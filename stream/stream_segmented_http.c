/*
 * Segmented parallel HTTP stream layer for mpv.
 *
 * Downloads the media file with N parallel HTTP Range connections
 * ("chunks"), each fetching a fixed-size segment ahead of the playback
 * position, and serves them to the demuxer as one seamless, seekable
 * byte stream.
 *
 * Enabled with:  --http-segmented-connections=N --http-segmented-chunk-size=SIZE
 * (N < 2 disables the module; the URL then falls through to the normal
 * stream_lavf single-connection path.)
 *
 * Next-file prefetch: a stream opened for playlist prefetch (STREAM_PREFETCH
 * flag, set by the player when --prefetch-playlist=immediate is on) starts with only
 * --http-segmented-prefetch-connections workers *active* — the readahead window and
 * buffers are still allocated at the full --http-segmented-connections size, but the
 * extra workers park until the file is promoted to the current playback
 * entry. On promotion the player sends STREAM_CTRL_SEGMENTED_ACTIVATE and all
 * workers wake. Workers and the readahead window are separate concepts here:
 * num_slots is the window (fixed at open); active_workers is download
 * parallelism (grows on promotion). This lets prefetch buffer the next file
 * cheaply — one connection, bounded by the demuxer's per-instance prefetch
 * cache cap — without stealing bandwidth from what is currently playing, then
 * ramp to full parallelism the instant it becomes the current file, with the
 * already-buffered data kept (no stream reopen).
 *
 * Fallback: if the server does not report byte-range/seek support, or
 * the file size is unknown, open() returns STREAM_NO_MATCH so that
 * stream_lavf takes over transparently.
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

#include <limits.h>
#include <string.h>

#include <libavformat/avformat.h>
#include <libavformat/avio.h>
#include <libavutil/dict.h>
#include <libavutil/opt.h>

#include "common/common.h"
#include "common/msg.h"
#include "demux/demux.h"
#include "misc/thread_tools.h"
#include "options/m_config.h"
#include "options/m_option.h"
#include "osdep/threads.h"
#include "osdep/timer.h"
#include "stream.h"

#include "mpv_talloc.h"

#define MAX_WORKERS 128

#define OPT_BASE_STRUCT struct segmented_http_opts
struct segmented_http_opts {
    int chunks;
    int64_t chunk_size;
    bool auto_size;
    int prefetch_chunks;
};

const struct m_sub_options stream_segmented_http_conf = {
    .opts = (const m_option_t[]) {
        {"http-segmented-connections", OPT_INT(chunks), M_RANGE(0, MAX_WORKERS)},
        {"http-segmented-chunk-size", OPT_BYTE_SIZE(chunk_size),
            M_RANGE(64 * 1024, (int64_t)1 << 30)},
        {"http-segmented-auto-size", OPT_BOOL(auto_size)},
        {"http-segmented-prefetch-connections", OPT_INT(prefetch_chunks),
            M_RANGE(1, MAX_WORKERS)},
        {0}
    },
    .size = sizeof(struct segmented_http_opts),
    .defaults = &(const struct segmented_http_opts){
        .chunks = 0,
        .chunk_size = 10 * 1024 * 1024,
        .auto_size = true,
        .prefetch_chunks = 1,
    },
};

enum slot_state {
    SLOT_EMPTY = 0,     // no chunk assigned, buffer free
    SLOT_QUEUED,        // chunk assigned, waiting for a worker
    SLOT_ACTIVE,        // a worker is downloading into it
    SLOT_DONE,          // download finished (possibly with ->error set)
    SLOT_ABANDONED,     // flushed while ACTIVE; worker still owns the buffer
};

struct slot {
    enum slot_state state;
    int64_t offset;     // absolute file offset of the chunk
    int64_t len;        // chunk length (last chunk may be short)
    int64_t filled;     // contiguous bytes downloaded so far
    int retries;        // failed download attempts for this chunk
    bool error;
    uint8_t *buf;       // capacity = p->chunk_size
};

struct worker {
    struct priv *p;
    stream_t *stream;
    int index;          // 0..num_slots-1
    mp_thread thread;
    bool thread_valid;
    AVIOContext *avio;
    int64_t avio_pos;   // stream offset the connection is positioned at
    struct slot *active;
    uint64_t dl_total;  // monotonic bytes this connection has downloaded
};

struct priv {
    // constants after open
    char *url;
    int num_slots;
    int64_t chunk_size;
    int64_t file_size;

    mp_mutex lock;
    mp_cond wakeup;

    // all following fields are protected by lock
    struct slot *slots;
    struct worker *workers;
    int num_workers;          // threads spawned (== num_slots)
    int active_workers;       // threads permitted to download (<= num_slots);
                              // < num_slots only while prefetching
    int64_t read_pos;         // next byte the reader wants
    bool shutdown;

    // download-speed sampling (see STREAM_CTRL_GET_SEGMENTED_SPEED); the EMA is
    // rolled on query, driven by the demuxer's ~1s update_cache() tick.
    int64_t speed_last_ns;
    int64_t last_dl_ns;       // last time any worker actually received bytes
    uint64_t worker_prev_total[MAX_WORKERS];
    double worker_bps[MAX_WORKERS];
};

static void schedule_slots(struct priv *p);

static char *seg_normalize_url(void *ta_parent, const char *filename)
{
    // Escape everything but reserved characters, don't double-escape.
    // Same normalization stream_lavf applies to http(s) URLs.
    return mp_url_escape(ta_parent, filename, ":/?#[]@!$&'()*+,;=%");
}

static int worker_interrupt_cb(void *ctx)
{
    struct worker *w = ctx;
    struct priv *p = w->p;
    if (mp_cancel_test(w->stream->cancel))
        return 1;
    mp_mutex_lock(&p->lock);
    bool stop = p->shutdown ||
                (w->active && w->active->state == SLOT_ABANDONED);
    mp_mutex_unlock(&p->lock);
    return stop;
}

// Open a new HTTP connection requesting exactly [off, end).
static int worker_open_conn(struct worker *w, int64_t off, int64_t end)
{
    struct priv *p = w->p;
    stream_t *s = w->stream;

    AVDictionary *dict = NULL;
    av_dict_set(&dict, "reconnect", "1", 0);
    av_dict_set(&dict, "reconnect_delay_max", "7", 0);
    av_dict_set(&dict, "multiple_requests", "1", 0);
    av_dict_set_int(&dict, "offset", off, 0);
    // Exact range: an open-ended request would keep the server pushing
    // data past the chunk into socket buffers, which gets discarded when
    // the connection is repositioned — pure wasted transfer. A bounded
    // range is fully consumed, which also lets ffmpeg reuse the
    // keep-alive connection for the next chunk instead of a new TLS setup.
    av_dict_set_int(&dict, "end_offset", end, 0);
    // ffmpeg's default "short seek" threshold is the TCP window size, so
    // it serves forward jumps up to several MB by reading and discarding
    // the gap — bytes another worker downloads anyway. Cap it so chunk
    // hops issue a new exact range request instead (still on the same
    // kept-alive connection).
    av_dict_set_int(&dict, "short_seek_size", 64 * 1024, 0);
    mp_setup_av_network_options(&dict, NULL, s->global, s->log);

    AVIOInterruptCB cb = {
        .callback = worker_interrupt_cb,
        .opaque = w,
    };

    int err = avio_open2(&w->avio, p->url, AVIO_FLAG_READ, &cb, &dict);
    av_dict_free(&dict);
    if (err < 0) {
        w->avio = NULL;
        return err;
    }
    w->avio_pos = off;
    return 0;
}

// Position the worker's connection to serve [off, end), (re)opening if
// needed. Seeking with an updated end_offset issues a new exact-range
// request; if the old response was fully drained ffmpeg reuses the
// keep-alive connection.
static int worker_position(struct worker *w, int64_t off, int64_t end)
{
    // avio_pos == off means the previous response ended exactly here; a
    // same-position avio_seek would no-op onto an exhausted request and
    // the next read would block, so force a fresh request instead.
    if (w->avio && w->avio_pos != off) {
        if (av_opt_set_int(w->avio, "end_offset", end,
                           AV_OPT_SEARCH_CHILDREN) >= 0 &&
            avio_seek(w->avio, off, SEEK_SET) == off)
        {
            w->avio_pos = off;
            return 0;
        }
    }
    if (w->avio)
        avio_closep(&w->avio);
    return worker_open_conn(w, off, end);
}

static MP_THREAD_VOID worker_thread(void *arg)
{
    struct worker *w = arg;
    struct priv *p = w->p;
    stream_t *s = w->stream;

    mp_mutex_lock(&p->lock);
    while (!p->shutdown) {
        // Parked while prefetching: this worker's index is beyond the active
        // set, so it must not open a connection or pull work. It wakes on the
        // broadcast from STREAM_CTRL_SEGMENTED_ACTIVATE (or shutdown).
        if (w->index >= p->active_workers) {
            mp_cond_wait(&p->wakeup, &p->lock);
            continue;
        }
        // pick the queued chunk closest to the reader
        struct slot *sl = NULL;
        for (int n = 0; n < p->num_slots; n++) {
            struct slot *c = &p->slots[n];
            if (c->state == SLOT_QUEUED && (!sl || c->offset < sl->offset))
                sl = c;
        }
        if (!sl) {
            mp_cond_wait(&p->wakeup, &p->lock);
            continue;
        }
        sl->state = SLOT_ACTIVE;
        w->active = sl;
        int64_t off = sl->offset;
        int64_t want = sl->len;
        mp_mutex_unlock(&p->lock);

        bool failed = false;
        int64_t got = 0;
        if (worker_position(w, off, off + want) < 0) {
            failed = true;
        } else {
            while (got < want) {
                int r = avio_read_partial(w->avio, sl->buf + got,
                                          MPMIN(want - got, 256 << 10));
                if (r <= 0) {
                    failed = true;
                    // force a fresh connection for the next chunk
                    avio_closep(&w->avio);
                    break;
                }
                got += r;
                w->avio_pos = off + got;

                mp_mutex_lock(&p->lock);
                if (p->shutdown || sl->state == SLOT_ABANDONED) {
                    // flushed under us: discard and release the slot
                    sl->state = SLOT_EMPTY;
                    sl->filled = 0;
                    w->active = NULL;
                    if (!p->shutdown)
                        schedule_slots(p);
                    mp_cond_broadcast(&p->wakeup);
                    mp_mutex_unlock(&p->lock);
                    goto next;
                }
                sl->filled = got;
                w->dl_total += r;   // per-connection throughput accounting
                p->last_dl_ns = mp_time_ns();
                mp_cond_broadcast(&p->wakeup);
                mp_mutex_unlock(&p->lock);
            }
        }

        mp_mutex_lock(&p->lock);
        if (sl->state == SLOT_ABANDONED) {
            sl->state = SLOT_EMPTY;
            sl->filled = 0;
            if (!p->shutdown)
                schedule_slots(p);
        } else if (failed && (p->shutdown || mp_cancel_test(s->cancel))) {
            // expected abort (quit/stop), not a download error
            sl->state = SLOT_EMPTY;
            sl->filled = 0;
        } else if (failed && sl->retries < 3) {
            sl->retries++;
            sl->filled = 0;
            sl->state = SLOT_QUEUED;
            MP_WARN(s, "segmented: retrying chunk at %"PRId64" "
                    "(attempt %d)\n", off, sl->retries + 1);
            mp_cond_broadcast(&p->wakeup);
        } else {
            sl->filled = got;
            sl->error = failed;
            sl->state = SLOT_DONE;
            if (failed) {
                MP_ERR(s, "segmented: chunk at %"PRId64" (%"PRId64" bytes) "
                       "failed after %"PRId64" bytes\n", off, want, got);
            }
        }
        w->active = NULL;
        mp_cond_broadcast(&p->wakeup);
        continue;

    next:
        mp_mutex_lock(&p->lock);
    }
    mp_mutex_unlock(&p->lock);

    if (w->avio)
        avio_closep(&w->avio);
    MP_THREAD_RETURN();
}

// Keep the window [read_pos, read_pos + num_slots * chunk_size) covered:
// reclaim fully consumed chunks and assign uncovered holes to free slots.
// Called with lock held.
static void schedule_slots(struct priv *p)
{
    int64_t lo = p->read_pos;
    int64_t hi = MPMIN(lo + (int64_t)p->num_slots * p->chunk_size,
                       p->file_size);
    bool queued = false;

    // reclaim chunks the reader has fully passed
    for (int n = 0; n < p->num_slots; n++) {
        struct slot *sl = &p->slots[n];
        if (sl->state == SLOT_DONE && sl->offset + sl->len <= lo) {
            sl->state = SLOT_EMPTY;
            sl->filled = 0;
        }
    }

    // walk the coverage forward from the read position, assigning each
    // hole to a free slot (chunk_size pieces, truncated at the next
    // already-covered range)
    int64_t cover = lo;
    while (cover < hi) {
        struct slot *cur = NULL;
        int64_t next_off = hi;
        for (int n = 0; n < p->num_slots; n++) {
            struct slot *sl = &p->slots[n];
            if (sl->state == SLOT_EMPTY || sl->state == SLOT_ABANDONED)
                continue;
            if (sl->offset <= cover && cover < sl->offset + sl->len) {
                cur = sl;
                break;
            }
            if (sl->offset > cover && sl->offset < next_off)
                next_off = sl->offset;
        }
        if (cur) {
            cover = cur->offset + cur->len;
            continue;
        }
        struct slot *free_sl = NULL;
        for (int n = 0; n < p->num_slots; n++) {
            if (p->slots[n].state == SLOT_EMPTY) {
                free_sl = &p->slots[n];
                break;
            }
        }
        if (!free_sl)
            break; // all buffers busy; retried when a slot is released
        free_sl->offset = cover;
        free_sl->len = MPMIN(p->chunk_size, next_off - cover);
        free_sl->filled = 0;
        free_sl->retries = 0;
        free_sl->error = false;
        free_sl->state = SLOT_QUEUED;
        cover += free_sl->len;
        queued = true;
    }

    if (queued)
        mp_cond_broadcast(&p->wakeup);
}

// The read position left the covered window (a seek). Drop only the chunks
// that cannot serve [pos, pos + window) — chunks that still can are kept,
// so seeking near or within the window doesn't discard downloaded data.
// Called with lock held.
static void re_anchor(struct priv *p, int64_t pos)
{
    int64_t hi = MPMIN(pos + (int64_t)p->num_slots * p->chunk_size,
                       p->file_size);
    for (int n = 0; n < p->num_slots; n++) {
        struct slot *sl = &p->slots[n];
        if (sl->state == SLOT_EMPTY || sl->state == SLOT_ABANDONED)
            continue;
        bool useful = sl->offset + sl->len > pos && sl->offset < hi &&
                      !(sl->state == SLOT_DONE && sl->error);
        if (useful)
            continue;
        if (sl->state == SLOT_ACTIVE) {
            // buffer still owned by the worker; it releases it on next check
            sl->state = SLOT_ABANDONED;
        } else {
            sl->state = SLOT_EMPTY;
            sl->filled = 0;
        }
    }

    // If nothing covers 'pos' and every buffer is occupied by kept chunks
    // (small backward seek into a full window), free the furthest-ahead
    // chunks until the gap at 'pos' has a slot to download into.
    while (1) {
        bool have_empty = false, covers_pos = false;
        struct slot *victim = NULL;
        for (int n = 0; n < p->num_slots; n++) {
            struct slot *sl = &p->slots[n];
            if (sl->state == SLOT_EMPTY) {
                have_empty = true;
            } else if (sl->state != SLOT_ABANDONED) {
                if (sl->offset <= pos && pos < sl->offset + sl->len)
                    covers_pos = true;
                if (!victim || sl->offset > victim->offset)
                    victim = sl;
            }
        }
        if (have_empty || covers_pos || !victim)
            break;
        if (victim->state == SLOT_ACTIVE) {
            victim->state = SLOT_ABANDONED;
        } else {
            victim->state = SLOT_EMPTY;
            victim->filled = 0;
        }
    }

    schedule_slots(p);
}

// Find the valid slot covering file position 'pos'. Lock held.
static struct slot *find_slot(struct priv *p, int64_t pos)
{
    for (int n = 0; n < p->num_slots; n++) {
        struct slot *sl = &p->slots[n];
        if (sl->state == SLOT_EMPTY || sl->state == SLOT_ABANDONED)
            continue;
        if (pos >= sl->offset && pos < sl->offset + sl->len)
            return sl;
    }
    return NULL;
}

static int seg_fill_buffer(stream_t *s, void *buffer, int max_len)
{
    struct priv *p = s->priv;
    int res = -1;

    mp_mutex_lock(&p->lock);
    while (1) {
        if (p->read_pos >= p->file_size) {
            res = 0; // EOF
            break;
        }
        if (mp_cancel_test(s->cancel))
            break;

        struct slot *sl = find_slot(p, p->read_pos);
        if (!sl) {
            // Window does not cover the read position (seek happened).
            // Re-anchor it here; if no slot could be scheduled (all are
            // still owned by workers draining abandoned chunks), drop the
            // lock and wait so the workers can release them — never spin
            // while holding the lock, or the workers can't make progress.
            re_anchor(p, p->read_pos);
            if (!find_slot(p, p->read_pos))
                mp_cond_timedwait(&p->wakeup, &p->lock, MP_TIME_MS_TO_NS(100));
            continue;
        }

        int64_t avail = sl->filled - (p->read_pos - sl->offset);
        if (avail > 0) {
            int n = MPMIN(avail, max_len);
            memcpy(buffer, sl->buf + (p->read_pos - sl->offset), n);
            p->read_pos += n;
            if (sl->state == SLOT_DONE && !sl->error &&
                p->read_pos >= sl->offset + sl->len)
            {
                sl->state = SLOT_EMPTY;
                sl->filled = 0;
                schedule_slots(p);
            }
            res = n;
            break;
        }

        if (sl->state == SLOT_DONE) {
            // error chunk with no more usable data
            break;
        }

        // wait for the worker, waking periodically to re-check cancel
        mp_cond_timedwait(&p->wakeup, &p->lock, MP_TIME_MS_TO_NS(100));
    }
    mp_mutex_unlock(&p->lock);
    return res;
}

static int seg_seek(stream_t *s, int64_t newpos)
{
    struct priv *p = s->priv;
    if (newpos < 0 || newpos > p->file_size)
        return 0;
    mp_mutex_lock(&p->lock);
    p->read_pos = newpos;
    // window adjustment happens lazily in fill_buffer
    mp_mutex_unlock(&p->lock);
    return 1;
}

static int64_t seg_get_size(stream_t *s)
{
    struct priv *p = s->priv;
    return p->file_size;
}

static int seg_control(stream_t *s, int cmd, void *arg)
{
    struct priv *p = s->priv;
    switch (cmd) {
    case STREAM_CTRL_SEGMENTED_ACTIVATE:
        // Promotion from next-file prefetch to the current playback entry:
        // wake the parked workers so download runs at full parallelism.
        mp_mutex_lock(&p->lock);
        if (p->active_workers < p->num_slots) {
            p->active_workers = p->num_slots;
            MP_VERBOSE(s, "segmented: activated full parallelism "
                       "(%d workers) on promotion\n", p->num_slots);
            mp_cond_broadcast(&p->wakeup);
        }
        mp_mutex_unlock(&p->lock);
        return STREAM_OK;
    case STREAM_CTRL_GET_SEGMENTED_SPEED: {
        // Report the true combined + per-connection download rate. This is the
        // sum of what the N worker sockets actually pull off the network, which
        // diverges from mpv's demuxer raw-input-rate: that measures the reader
        // draining already-downloaded slot buffers (a memcpy), not network I/O.
        struct stream_segmented_speed *out = arg;
        mp_mutex_lock(&p->lock);
        int64_t now = mp_time_ns();
        int64_t dt = now - p->speed_last_ns;
        // Hard-zero once the workers have gone idle (window full, or the whole
        // file downloaded): without this the last EMA sample would linger on
        // screen because nothing re-queries us once caching stops.
        if (p->last_dl_ns && (now - p->last_dl_ns) > MP_TIME_MS_TO_NS(1500)) {
            for (int n = 0; n < p->num_slots; n++) {
                p->worker_bps[n] = 0;
                p->worker_prev_total[n] = p->workers[n].dl_total;
            }
            p->speed_last_ns = now;
        } else if (!p->speed_last_ns || dt >= MP_TIME_S_TO_NS(1)) {
            double secs = p->speed_last_ns
                ? dt / (double)MP_TIME_S_TO_NS(1) : 1.0;
            for (int n = 0; n < p->num_slots; n++) {
                uint64_t cur = p->workers[n].dl_total;
                uint64_t delta = cur - p->worker_prev_total[n];
                p->worker_prev_total[n] = cur;
                double inst = delta / secs;
                // 50/50 EMA, matching demux.c's bytes_per_second smoothing
                p->worker_bps[n] = 0.5 * p->worker_bps[n] + 0.5 * inst;
            }
            p->speed_last_ns = now;
        }
        // Report only the workers that are actually permitted to download:
        // active_workers == num_slots for the current file, but
        // == --http-segmented-prefetch-connections while a stream is still
        // prefetching, so the number of per-thread rates tracks whichever
        // count is in effect.
        uint64_t total = 0;
        for (int n = 0; n < p->active_workers; n++) {
            out->worker_bps[n] = (uint64_t)p->worker_bps[n];
            total += (uint64_t)p->worker_bps[n];
        }
        out->num_workers = p->active_workers;
        out->total_bps = total;
        mp_mutex_unlock(&p->lock);
        return STREAM_OK;
    }
    }
    return STREAM_UNSUPPORTED;
}

static void seg_close(stream_t *s)
{
    struct priv *p = s->priv;
    if (!p)
        return;
    mp_mutex_lock(&p->lock);
    p->shutdown = true;
    mp_cond_broadcast(&p->wakeup);
    mp_mutex_unlock(&p->lock);
    for (int n = 0; n < p->num_workers; n++) {
        if (p->workers[n].thread_valid)
            mp_thread_join(p->workers[n].thread);
    }
    mp_cond_destroy(&p->wakeup);
    mp_mutex_destroy(&p->lock);
}

// Reconcile chunks*size with demuxer-max-bytes as requested:
//  - grow segment size (bounded) if demuxer-max-bytes leaves headroom
//  - raise demuxer-max-bytes if the window would exceed it
// While prefetching we never raise the global demuxer-max-bytes: that write
// would be seen as a demuxer option change and make the player drop the very
// prefetch being built. The per-instance prefetch cache cap governs the
// prefetch volume anyway, and the global limit is restored/authoritative once
// the file is promoted.
static void reconcile_demux_buffer(stream_t *s, struct segmented_http_opts *o,
                                   int64_t *chunk_size)
{
    struct m_config_cache *dc =
        m_config_cache_alloc(s, s->global, &demux_conf);
    if (!dc)
        return;
    struct demux_opts *dopts = dc->opts;

    int64_t total = (int64_t)o->chunks * *chunk_size;

    if (o->auto_size && total < dopts->max_bytes) {
        // grow chunks toward the demuxer budget, but never more than 4x the
        // requested size and never past 1 GiB total window
        int64_t target = MPMIN(dopts->max_bytes / o->chunks, *chunk_size * 4);
        target = MPMIN(target, ((int64_t)1 << 30) / o->chunks);
        if (target > *chunk_size) {
            MP_VERBOSE(s, "segmented: growing chunk-size %"PRId64" -> "
                       "%"PRId64" to target demuxer-max-bytes (%"PRId64")\n",
                       *chunk_size, target, (int64_t)dopts->max_bytes);
            *chunk_size = target;
            total = (int64_t)o->chunks * *chunk_size;
        }
    }

    if (total > dopts->max_bytes && !s->prefetch) {
        MP_INFO(s, "segmented: raising demuxer-max-bytes %"PRId64" -> "
                "%"PRId64" to fit %d x %"PRId64" window\n",
                (int64_t)dopts->max_bytes, total, o->chunks, *chunk_size);
        dopts->max_bytes = total;
        m_config_cache_write_opt(dc, &dopts->max_bytes);
    }
}

static int seg_open(stream_t *stream)
{
    if (stream->mode != STREAM_READ)
        return STREAM_NO_MATCH;

    struct segmented_http_opts *opts =
        mp_get_config_group(stream, stream->global, &stream_segmented_http_conf);

    if (opts->chunks < 2)
        return STREAM_NO_MATCH; // disabled -> normal stream_lavf path

    char *url = seg_normalize_url(stream, stream->url);

    // Probe connection: verify range/seek support and get the file size.
    AVDictionary *dict = NULL;
    av_dict_set(&dict, "reconnect", "1", 0);
    av_dict_set(&dict, "reconnect_delay_max", "7", 0);
    // The probe only needs headers (size, ranges, mime); bound the range
    // so the server doesn't stream the whole file at us until the close
    // propagates. Size/seekability still come from Content-Range and
    // Accept-Ranges of the bounded response.
    av_dict_set_int(&dict, "end_offset", 256 * 1024, 0);
    mp_setup_av_network_options(&dict, NULL, stream->global, stream->log);

    struct priv *p = talloc_zero(stream, struct priv);
    stream->priv = p;

    // temporary single-shot interrupt context for the probe
    struct worker probe = { .p = p, .stream = stream };
    AVIOInterruptCB cb = {
        .callback = worker_interrupt_cb,
        .opaque = &probe,
    };

    mp_mutex_init(&p->lock);
    mp_cond_init(&p->wakeup);

    AVIOContext *avio = NULL;
    int err = avio_open2(&avio, url, AVIO_FLAG_READ, &cb, &dict);
    av_dict_free(&dict);
    if (err < 0) {
        MP_VERBOSE(stream, "segmented: probe open failed (%s)\n",
                   av_err2str(err));
        goto fallback;
    }

    int64_t size = avio_size(avio);
    bool seekable = avio->seekable & AVIO_SEEKABLE_NORMAL;

    if (avio->av_class) {
        uint8_t *mt = NULL;
        if (av_opt_get(avio, "mime_type", AV_OPT_SEARCH_CHILDREN, &mt) >= 0) {
            stream->mime_type = talloc_strdup(stream, mt);
            av_free(mt);
        }
    }
    avio_closep(&avio);

    if (!seekable || size <= 0) {
        MP_INFO(stream, "segmented: server lacks range support or size "
                "unknown; falling back to normal streaming\n");
        goto fallback;
    }

    int64_t chunk_size = opts->chunk_size;
    reconcile_demux_buffer(stream, opts, &chunk_size);

    // no point holding more window than file
    int num_slots = opts->chunks;
    while (num_slots > 1 && (int64_t)(num_slots - 1) * chunk_size >= size)
        num_slots--;

    // Download parallelism starts reduced for a next-file prefetch stream and
    // grows to num_slots on promotion (STREAM_CTRL_SEGMENTED_ACTIVATE); the
    // window (num_slots) and buffers are always allocated at full size so no
    // reallocation is ever needed while workers hold slot pointers.
    int active = num_slots;
    if (stream->prefetch)
        active = MPCLAMP(opts->prefetch_chunks, 1, num_slots);

    p->url = talloc_strdup(p, url);
    p->num_slots = num_slots;
    p->chunk_size = chunk_size;
    p->file_size = size;
    p->active_workers = active;
    p->slots = talloc_zero_array(p, struct slot, num_slots);
    for (int n = 0; n < num_slots; n++)
        p->slots[n].buf = talloc_size(p, chunk_size);

    p->workers = talloc_zero_array(p, struct worker, num_slots);
    p->num_workers = num_slots;

    mp_mutex_lock(&p->lock);
    schedule_slots(p);
    mp_mutex_unlock(&p->lock);

    for (int n = 0; n < num_slots; n++) {
        struct worker *w = &p->workers[n];
        w->p = p;
        w->stream = stream;
        w->index = n;
        if (mp_thread_create(&w->thread, worker_thread, w)) {
            MP_ERR(stream, "segmented: could not spawn worker %d\n", n);
            seg_close(stream);
            return STREAM_ERROR;
        }
        w->thread_valid = true;
    }

    MP_INFO(stream, "segmented: %d parallel chunks x %"PRId64" bytes "
            "(window %"PRId64" MiB, file %"PRId64" MiB)%s\n",
            num_slots, chunk_size,
            (int64_t)num_slots * chunk_size / (1024 * 1024),
            size / (1024 * 1024),
            active < num_slots ? " [prefetch: reduced parallelism]" : "");
    if (active < num_slots) {
        MP_VERBOSE(stream, "segmented: prefetch mode, %d of %d workers active "
                   "until promotion\n", active, num_slots);
    }

    stream->fill_buffer = seg_fill_buffer;
    stream->seek = seg_seek;
    stream->seekable = true;
    stream->get_size = seg_get_size;
    stream->control = seg_control;
    stream->close = seg_close;
    stream->streaming = true;
    stream->is_network = true;
    return STREAM_OK;

fallback:
    mp_cond_destroy(&p->wakeup);
    mp_mutex_destroy(&p->lock);
    stream->priv = NULL;
    return STREAM_NO_MATCH;
}

const stream_info_t stream_info_segmented_http = {
    .name = "segmented",
    .open = seg_open,
    .protocols = (const char *const[]){"http", "https", NULL},
    .stream_origin = STREAM_ORIGIN_NET,
};
