/*
 * In-process, from-buffer thumbnailer -- command handler.
 *
 * Implements the "thumbnail-cache" command:
 *
 *     thumbnail-cache <time> <width> <height> <filename>
 *
 * For network (or non-seekable) sources, it asks the demuxer for video
 * packets that are ALREADY in the seekable cache around <time> (see
 * demux_get_cached_video_packets(); no network I/O, no disturbance to
 * playback). For local seekable files, it opens the file directly with a
 * private libavformat context (kept open and reused across calls) and can
 * seek anywhere. Either way it decodes one frame with a private libavcodec
 * context, scales it to exactly <width>x<height> BGRA, and writes the raw
 * pixels (tightly packed, width*height*4 bytes) to <filename> atomically.
 *
 * The output format matches what thumbnail scripts feed to overlay-add.
 * cmd->success is set true only when a thumbnail file was written.
 *
 * This file is part of mpv.
 *
 * mpv is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * mpv is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with mpv.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/frame.h>
#include <libavutil/rational.h>

#include "mpv_talloc.h"
#include "common/av_common.h"
#include "common/common.h"
#include "demux/demux.h"
#include "demux/packet.h"
#include "demux/stheader.h"
#include "input/cmd.h"
#include "osdep/threads.h"
#include "video/img_format.h"
#include "video/mp_image.h"
#include "video/sws_utils.h"
#include "core.h"
#include "command.h"
#include "thumbnail.h"

// libavcodec pts timebase we impose on both packets and the decoded frame so
// timestamps are directly comparable to the requested time (in seconds).
#define THUMB_TB ((AVRational){1, AV_TIME_BASE})

// Decode the packet list and pick the frame at/nearest (but not after) the
// requested pts. Returns a fresh AVFrame ref on success (caller frees), or NULL.
static AVFrame *decode_target_frame(AVCodecParameters *avp,
                                    struct demux_packet **pkts, int npkts,
                                    double target)
{
    const AVCodec *dec = avcodec_find_decoder(avp->codec_id);
    if (!dec)
        return NULL;

    AVCodecContext *avctx = avcodec_alloc_context3(dec);
    if (!avctx)
        return NULL;

    AVFrame *best = NULL, *frame = NULL;
    AVPacket *apkt = NULL;
    AVRational tb = THUMB_TB;
    int64_t target_ts = (int64_t)(target * AV_TIME_BASE);

    if (avcodec_parameters_to_context(avctx, avp) < 0)
        goto done;

    // Favour speed -- thumbnails don't need loop filtering or slice threads.
    avctx->pkt_timebase = THUMB_TB;
    avctx->flags2 |= AV_CODEC_FLAG2_FAST;
    avctx->skip_loop_filter = AVDISCARD_ALL;
    avctx->thread_count = 1;

    if (avcodec_open2(avctx, dec, NULL) < 0)
        goto done;

    frame = av_frame_alloc();
    apkt = av_packet_alloc();
    if (!frame || !apkt)
        goto done;

    bool stop = false;
    for (int i = 0; i <= npkts && !stop; i++) {
        // i == npkts: flush the decoder (send NULL) to drain delayed frames.
        if (i < npkts) {
            mp_set_av_packet(apkt, pkts[i], &tb);
            if (avcodec_send_packet(avctx, apkt) < 0)
                break;
        } else {
            if (avcodec_send_packet(avctx, NULL) < 0)
                break;
        }

        while (true) {
            int r = avcodec_receive_frame(avctx, frame);
            if (r == AVERROR(EAGAIN) || r == AVERROR_EOF)
                break;
            if (r < 0)
                goto done;

            int64_t fts = frame->best_effort_timestamp;
            if (fts == AV_NOPTS_VALUE)
                fts = frame->pts;

            if (!best) {
                best = av_frame_alloc();
                av_frame_move_ref(best, frame);
            } else if (fts == AV_NOPTS_VALUE || fts <= target_ts) {
                av_frame_unref(best);
                av_frame_move_ref(best, frame);
            } else {
                // We've passed the target and already have a good frame.
                av_frame_unref(frame);
                stop = true;
                break;
            }
            av_frame_unref(frame);
        }
    }

done:
    // apkt->buf/side_data are BORROWED from the source packets (mp_set_av_packet
    // does not take a ref); clear them before freeing so we don't unref buffers
    // still owned by the demuxer cache. This is exactly what mp_free_av_packet does.
    mp_free_av_packet(&apkt);
    av_frame_free(&frame);
    avcodec_free_context(&avctx);
    return best;
}

// Scale to exactly out_w x out_h BGRA and write tightly-packed pixels to path,
// via a temporary file + rename so a concurrent reader never sees a partial.
static bool scale_and_write(AVFrame *av_frame,
                            int out_w, int out_h, const char *path)
{
    bool ok = false;
    struct mp_image *src = NULL, *dst = NULL;
    struct mp_sws_context *sws = NULL;
    char *tmp = NULL;
    FILE *f = NULL;

    src = mp_image_from_av_frame(av_frame);
    if (!src)
        goto done;

    dst = mp_image_alloc(IMGFMT_BGRA, out_w, out_h);
    if (!dst)
        goto done;

    sws = mp_sws_alloc(NULL);
    if (!sws)
        goto done;
    sws->allow_zimg = false;
    if (mp_sws_scale(sws, dst, src) < 0)
        goto done;

    tmp = talloc_asprintf(NULL, "%s.writing", path);
    f = fopen(tmp, "wb");
    if (!f)
        goto done;

    for (int y = 0; y < out_h; y++) {
        if (fwrite(dst->planes[0] + (ptrdiff_t)y * dst->stride[0], 1,
                   (size_t)out_w * 4, f) != (size_t)out_w * 4)
            goto done;
    }
    fclose(f);
    f = NULL;

    if (rename(tmp, path) != 0) {
        remove(tmp);
        goto done;
    }
    ok = true;

done:
    if (f)
        fclose(f);
    talloc_free(tmp);
    talloc_free(sws);
    talloc_free(dst);
    talloc_free(src);
    return ok;
}

// ---------------------------------------------------------------------------
// Local-file path: for seekable non-network sources the whole timeline is on
// disk, so instead of the (windowed) demuxer cache we open the file directly
// and seek anywhere. A private libavformat context is kept open and reused
// across hovers (reopening/probing per request would be wasteful); it is
// rebuilt when the played file changes. The state is per-player
// (mpctx->thumbnail_ctx - a process can host multiple libmpv instances) and
// serialized by its lock, so concurrent worker threads don't touch the
// context at once.
// ---------------------------------------------------------------------------

struct thumbnail_ctx {
    mp_mutex lock;
    // all following fields are protected by lock
    char *path;                 // currently open file
    AVFormatContext *fmt;
    AVCodecContext *avctx;
    int vstream;
};

// ft_lock held.
static void ft_close(struct thumbnail_ctx *ctx)
{
    if (ctx->avctx)
        avcodec_free_context(&ctx->avctx);
    if (ctx->fmt)
        avformat_close_input(&ctx->fmt);
    TA_FREEP(&ctx->path);
    ctx->vstream = -1;
}

// Open filename and set up the video decoder. ctx->lock held. Returns success.
static bool ft_open(struct thumbnail_ctx *ctx, const char *filename)
{
    ft_close(ctx);

    AVFormatContext *fmt = NULL;
    if (avformat_open_input(&fmt, filename, NULL, NULL) < 0)
        return false;
    if (avformat_find_stream_info(fmt, NULL) < 0)
        goto fail;

    int vs = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (vs < 0)
        goto fail;

    const AVCodec *dec = avcodec_find_decoder(fmt->streams[vs]->codecpar->codec_id);
    if (!dec)
        goto fail;

    AVCodecContext *avctx = avcodec_alloc_context3(dec);
    if (!avctx)
        goto fail;
    if (avcodec_parameters_to_context(avctx, fmt->streams[vs]->codecpar) < 0) {
        avcodec_free_context(&avctx);
        goto fail;
    }
    avctx->pkt_timebase = fmt->streams[vs]->time_base;
    avctx->flags2 |= AV_CODEC_FLAG2_FAST;
    avctx->skip_loop_filter = AVDISCARD_ALL;
    avctx->thread_count = 1;
    if (avcodec_open2(avctx, dec, NULL) < 0) {
        avcodec_free_context(&avctx);
        goto fail;
    }

    ctx->fmt = fmt;
    ctx->avctx = avctx;
    ctx->vstream = vs;
    ctx->path = talloc_strdup(ctx, filename);
    return true;

fail:
    avformat_close_input(&fmt);
    return false;
}

static bool thumb_from_file(struct thumbnail_ctx *ctx, const char *filename,
                            double target, int out_w, int out_h,
                            const char *path)
{
    bool ok = false;
    mp_mutex_lock(&ctx->lock);

    if (!ctx->path || strcmp(ctx->path, filename) != 0) {
        if (!ft_open(ctx, filename))
            goto done;
    }

    AVStream *st = ctx->fmt->streams[ctx->vstream];
    int64_t ts = (int64_t)(target / av_q2d(st->time_base));
    if (st->start_time != AV_NOPTS_VALUE)
        ts += st->start_time;

    // Seek to the keyframe at/before the target (same keyframe-accurate result
    // as the network cache path), then decode the first frame it yields.
    if (av_seek_frame(ctx->fmt, ctx->vstream, ts, AVSEEK_FLAG_BACKWARD) < 0)
        goto done;
    avcodec_flush_buffers(ctx->avctx);

    AVPacket *pkt = av_packet_alloc();
    AVFrame *frame = av_frame_alloc();
    bool have_frame = false;
    if (pkt && frame) {
        while (true) {
            if (av_read_frame(ctx->fmt, pkt) < 0)
                break;
            if (pkt->stream_index != ctx->vstream) {
                av_packet_unref(pkt);
                continue;
            }
            int sr = avcodec_send_packet(ctx->avctx, pkt);
            av_packet_unref(pkt);
            if (sr < 0)
                break;
            int r = avcodec_receive_frame(ctx->avctx, frame);
            if (r == AVERROR(EAGAIN))
                continue;           // decoder wants more packets
            if (r < 0)
                break;
            have_frame = true;      // first frame after the keyframe seek
            break;
        }
    }
    if (have_frame)
        ok = scale_and_write(frame, out_w, out_h, path);
    av_frame_free(&frame);
    av_packet_free(&pkt);

done:
    mp_mutex_unlock(&ctx->lock);
    return ok;
}

void mp_thumbnail_uninit(struct MPContext *mpctx)
{
    struct thumbnail_ctx *ctx = mpctx->thumbnail_ctx;
    if (!ctx)
        return;
    ft_close(ctx);
    mp_mutex_destroy(&ctx->lock);
    talloc_free(ctx);
    mpctx->thumbnail_ctx = NULL;
}

// Network path: decode the keyframe out of the already-buffered demuxer cache.
static bool thumb_from_cache(struct MPContext *mpctx, double time,
                             int out_w, int out_h, const char *path)
{
    struct demuxer *demuxer = mpctx->demuxer;

    int vindex = -1;
    AVCodecParameters *avp = NULL;
    for (int i = 0; i < demux_get_num_stream(demuxer); i++) {
        struct sh_stream *sh = demux_get_stream(demuxer, i);
        if (sh && sh->type == STREAM_VIDEO && demux_stream_is_selected(sh)) {
            vindex = sh->index;
            avp = mp_codec_params_to_av(sh->codec);
            break;
        }
    }
    if (vindex < 0 || !avp) {
        if (avp)
            avcodec_parameters_free(&avp);
        return false;               // stays core-locked; caller returns locked
    }

    struct demux_packet **pkts = NULL;
    int npkts = 0;
    bool have = demux_get_cached_video_packets(demuxer, vindex, time,
                                               &pkts, &npkts);

    mp_core_unlock(mpctx);

    bool ok = false;
    if (have && npkts > 0) {
        AVFrame *frame = decode_target_frame(avp, pkts, npkts, time);
        if (frame) {
            ok = scale_and_write(frame, out_w, out_h, path);
            av_frame_free(&frame);
        }
    }

    for (int i = 0; i < npkts; i++)
        free_demux_packet(pkts[i]);
    talloc_free(pkts);
    avcodec_parameters_free(&avp);

    mp_core_lock(mpctx);
    return ok;
}

void cmd_thumbnail_cache(void *p)
{
    struct mp_cmd_ctx *cmd = p;
    struct MPContext *mpctx = cmd->mpctx;

    double time  = cmd->args[0].v.d;
    int    out_w = cmd->args[1].v.i;
    int    out_h = cmd->args[2].v.i;
    char  *path  = cmd->args[3].v.s;

    cmd->success = false;

    if (out_w <= 0 || out_h <= 0 || !path || !path[0])
        return;

    // --- core-locked: decide source and copy what we need to use off-lock ---
    struct demuxer *demuxer = mpctx->demuxer;
    if (!demuxer)
        return;

    // Local seekable file: read straight from disk (whole timeline available).
    // Network (or non-seekable): read from the buffered demuxer cache.
    bool local = !demuxer->is_network && demuxer->seekable &&
                 demuxer->filename && demuxer->filename[0];
    char *filename = local ? talloc_strdup(NULL, demuxer->filename) : NULL;

    // Lazily allocated; safe because the core lock is held here (concurrent
    // thumbnail-cache commands serialize on it before their worker phase).
    if (local && !mpctx->thumbnail_ctx) {
        struct thumbnail_ctx *ctx = talloc_zero(NULL, struct thumbnail_ctx);
        mp_mutex_init(&ctx->lock);
        ctx->vstream = -1;
        mpctx->thumbnail_ctx = ctx;
    }

    bool ok;
    if (local) {
        struct thumbnail_ctx *ctx = mpctx->thumbnail_ctx;
        mp_core_unlock(mpctx);
        ok = thumb_from_file(ctx, filename, time, out_w, out_h, path);
        talloc_free(filename);
        mp_core_lock(mpctx);
    } else {
        ok = thumb_from_cache(mpctx, time, out_w, out_h, path);
    }

    cmd->success = ok;
}
