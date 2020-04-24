import argparse
import cv2
import threading

from chunk import Chunk
from cache import Cache
from utils import ChunkC3DExtractor, ChunkVGGExtractor, StoppableThread

running_threads = {}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_name', type=str, default='./out.mp4')
    parser.add_argument('--database_name', type=str, default='vqadb')

    parser.add_argument('--cache_size', type=int, default='100') # in terms of chunks
    parser.add_argument('--chunk_size', type=int, default='10') # in terms of frames
    parser.add_argument('--evict_mod', type=int, default='10') # how often to evict (in terms of chunks)
    parser.add_argument('--use_ram', type=bool, default='True') # use RAM for cache

    # Not necessary?
    parser.add_argument('--database_user', type=str, default='vqadbuser')
    parser.add_argument('--database_password', type=str, default='pwd')

    args = parser.parse_args()
    return args    


def process_video(video_name, chunk_size, cache, c3d_extractor, vgg_extractor):
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
        cache, chunk_count, c3d_extracor, vgg_extractor
    )

    while True:
        flag, frame = cap.read()
        if flag:
            pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

            if frame_count == chunk_size:
                running_threads.update({
                    chunk_count: current_chunk.commit()
                })
                current_chunk = Chunk(
                    cache, chunk_count, c3d_extracor, vgg_extractor
                )
                chunk_count += 1
            else:
                current_chunk.add_frame(pos_frame, frame_count)

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
    c3d_extractor = ChunkC3DExtractor(args.chunk_size)
    vgg_extractor = ChunkVGGExtractor(args.chunk_size)
    

    # Run threads
    process_video_thread = StoppableThread(
    threading.Thread(target=process_video, args=(
        args.video_name, args.chunk_size, cache, c3d_extractor, vgg_extractor
        ))
    )
    ask_questions_thread = StoppableThread(
    threading.Thread(target=ask_questions, args=())
    )
    kill_old_threads_thread = StoppableThread(
        threading.Thread(target=kill_old_threads, args=(cache,))
    )

    print("Threads started.")


main()