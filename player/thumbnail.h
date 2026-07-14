/*
 * In-process, from-buffer thumbnailer -- command entry point.
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
#ifndef MP_PLAYER_THUMBNAIL_H
#define MP_PLAYER_THUMBNAIL_H

struct MPContext;

// Handler for the "thumbnail-cache" command. Runs on a worker thread
// (mp_cmd_def.spawn_thread).
void cmd_thumbnail_cache(void *p);

// Free the per-player thumbnailer state (kept file/decoder contexts), if any.
void mp_thumbnail_uninit(struct MPContext *mpctx);

#endif
