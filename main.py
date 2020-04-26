import argparse
import cv2
import time
import threading

from chunk import Chunk
from cache import Cache
from utils import StoppableThread

running_threads = {}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_name', type=str, default='./out.mp4')
    parser.add_argument('--database_name', type=str, default='vqadb')

    ## CACHE VARIABLES ##
    # Cache size (in terms of chunks)
    parser.add_argument('--cache_size', type=int, default='100')
    # How often to evict (in terms of chunks). A lower number 
    # (more frequent evictions) means the chunk creation and 
    # cache insertion process takes longer.
    parser.add_argument('--evict_mod', type=int, default='10')
    # Whether to use RAM for the cache
    parser.add_argument('--use_ram', type=bool, default='True')

    ## VIDEO PROCESSING VARIABLES ##
    # The size of a video chunk to cache
    parser.add_argument('--chunk_size', type=int, default='10')

    ## C3D FEATURE EXTRACTION ##
    # Depending on how these variables are set, some video data
    # might be lost. A chunk is divided into evenly-spaced clips
    # of some fixed size. If clip frames don't overlap then video
    # data is lost. Likewise, if clips frames overlap too much,
    # potentially too much computation occurs with no benefit.

    # Amount of clips to create per chunk. Can be <= chunk_size
    parser.add_argument('--clip_num_c3d', type=int, default='5')
    # Frames per clip. Can be <= chunk_size
    parser.add_argument('--frames_per_clip_c3d', type=int, default='2')

    # Not necessary?
    parser.add_argument('--database_user', type=str, default='vqadbuser')
    parser.add_argument('--database_password', type=str, default='pwd')

    args = parser.parse_args()
    return args    


def process_video(
    video_name, chunk_size, cache, frames_per_clip_c3d, clip_num_c3d
):
    # TODO: LATER-- replace this with something more real-time
    # and not frame-by-frame
    cap = cv2.VideoCapture(video_name)
    while not cap.isOpened():
        cap = cv2.VideoCapture(video_name)
        cv2.waitKey(1000)

    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    chunk_count = 0
    frame_count = 0
    current_chunk = Chunk(
        cache, chunk_count, chunk_size, frames_per_clip_c3d, clip_num_c3d
    )

    while True:
        flag, frame = cap.read()
        if flag:
            pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            
            print("processing frame")
            if frame_count == chunk_size:
                running_threads.update({
                    chunk_count: current_chunk.commit()
                })
                current_chunk = Chunk(
                    cache, chunk_count, chunk_size, frames_per_clip
                )
                chunk_count += 1
            else:
                current_chunk.add_frame(frame, frame_count)

            frame_count += 1
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
            print("frame is not ready")
            cv2.waitKey(1000)

        if cv2.waitKey(10) == 27:
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            break
    pass


def ask_questions():
    # TODO: write question-asking code here when ready
    pass


def kill_old_threads(cache):
    while True:
        ids_to_kill = [(k, v) for k, v in running_threads.items() if k < cache.oldest_id]
        for id in ids_to_kill:
            running_threads.get(id).stop()

        time.sleep(10)


def main():
    args = parse_args()

    cache = Cache(
        args.database_name,
        args.cache_size,
        args.evict_mod,
        args.use_ram,
    )
    
    # Run threads
    process_video_thread = threading.Thread(
      target=process_video, args=(
          args.video_name,
          args.chunk_size,
          cache,
          args.frames_per_clip_c3d,
          args.clip_num_c3d
        )
    )
    ask_questions_thread = threading.Thread(
        target=ask_questions, args=()
    )
    kill_old_threads_thread = threading.Thread(
        target=kill_old_threads, args=(cache,)
    )

    process_video_thread.start()
    ask_questions_thread.start()
    kill_old_threads_thread.start()

    while True:
        continue


main()
