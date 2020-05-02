import argparse
import cv2
import time
import threading
import sys
import logging
import random
import os

from chunk import Chunk
from cache import Cache
from chunk_localization import Chunk_Localization
from vqa import VQA
from other import StoppableThread


running_threads = {}


def parse_args():
    parser = argparse.ArgumentParser()
    # Enter either 'video_dir' or 'video_name'.
    # Using a video_dir will run a random interleaving of videos in
    # the directory, using a video_name will just run the video.
    parser.add_argument(
        '--video_dir', type=str,  default='VideoQA/MSVD-QA/video/'
    )
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

    ## VIDEO QA VARIABLES ##
    # "Config_id" for VideoQA in config.py
    parser.add_argument('--videoqa_config', type=str, default='0')
    # Path to the VideoQA model. This default is what generates when
    # you run the run_gra command in the VideoQA directory. If you train
    # the model by running a different VideoQA command, this might be in
    # a different location.
    parser.add_argument(
        '--videoqa_model_path', type=str, default='VideoQA/log/evqa'
    )
    parser.add_argument(
        '--videoqa_vocab_path', type=str, 
        default='VideoQA/data/msvd_qa/vocab.txt'
    )

    # TMLGA VARIABLES ##
    parser.add_argument('--config_file_path', type=str, default='0')
    parser.add_argument(
        '--vocab_file_path', type=str, 
        default='TMLGA/charades_vocab_1_30.pickle'
    )
    parser.add_argument(
        '--embeddings_file_path', type=str,
        default="TMLGA/charades_embeddings_1_30.pth"
    )
    parser.add_argument(
        '--i3d_extractor_model_path', type=str,
        default="pytorch_i3d/models/rgb_imagenet.pt"
    )
    parser.add_argument('--max_question_length', type=int, default='30')
    parser.add_argument('--min_question_length', type=int, default='3')


    args = parser.parse_args()
    return args    

def _open_video(video_name):
    cap = cv2.VideoCapture(video_name)
    while not cap.isOpened():
        cap = cv2.VideoCapture(video_name)
        cv2.waitKey(1000)
    return cap


def process_video(
    chunk_size, cache, 
    frames_per_clip_c3d, clip_num_c3d,
    i3d_extractor_model_path,
    video_name=False,
    video_dir=False
):

    # TODO: LATER-- replace this with something more real-time
    # and not frame-by-frame
    if video_name:
        cap = _open_video(video_name)
    else:
        interleaving = _create_video_interleaving(video_dir)
        cap = _open_video(video_dir + "/" + interleaving[0])

    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    
    chunk_count = 0
    frame_count = 0
    video_count = 0
    current_chunk = Chunk(
        cache, chunk_count, chunk_size,
        frames_per_clip_c3d, clip_num_c3d,
        i3d_extractor_model_path
    )
    import pdb; pdb.set_trace()
    while True:
        flag, frame = cap.read()
        if flag:
            pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            
            logging.info("processing frame #" + str(pos_frame))
            if frame_count == chunk_size:
                logging.debug("finishing chunk #" + str(chunk_count))
                # TODO: Run this in a new thread and not concurrently
                running_threads.update({
                    chunk_count: current_chunk.commit()
                })
                chunk_count += 1
                frame_count = 0
                current_chunk = Chunk(
                    cache, chunk_count, chunk_size,
                    frames_per_clip_c3d, clip_num_c3d,
                    i3d_extractor_model_path
                )
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
            video_count += 1
            if video_count < len(interleaving):
                cap = _open_video(video_dir + "/" + interleaving[video_count])
            else:
                break


def kill_old_threads(cache):
    while True:
        ids_to_kill = [(k, v) for k, v in running_threads.items() if k < cache.oldest_id]
        for id in ids_to_kill:
            running_threads.get(id).stop()

        time.sleep(10)


def _create_video_interleaving(video_directory):
    interleaving = []
    for f in os.listdir(video_directory):
      interleaving.append(f)
    
    random.shuffle(interleaving)
    return interleaving


def main():
    logging.basicConfig(filename='VQA.log',level=logging.DEBUG)
    args = parse_args()
    
    cache = Cache(
        args.database_name,
        args.cache_size,
        args.evict_mod,
        args.use_ram,
    )

    #chunk_localization = Chunk_Localization(
    #    args.config_file_path,
    #    args.vocab_file_path,
    #    args.embeddings_file_path,
    #    args.max_question_length,
    #    args.min_question_length,
    #    args.chunk_size
    #)

    print("Setup done")
    
    # Run threads
    #process_video_thread = threading.Thread(
    #  target=process_video, args=(
    #      args.chunk_size,
    #      cache,
    #      args.frames_per_clip_c3d,
    #      args.clip_num_c3d,
    #      args.i3d_extractor_model_path,
    #      video_dir=args.video_dir,
    #    )
    #)
    #process_video_thread.daemon = True

    kill_old_threads_thread = threading.Thread(
        target=kill_old_threads, args=(cache,)
    )

    #process_video_thread.start()
    kill_old_threads_thread.start()
    
    process_video(
            args.chunk_size,
            cache,
            args.frames_per_clip_c3d,
            args.clip_num_c3d,
            args.i3d_extractor_model_path,
            video_dir=args.video_dir
    )
#    while True:
#        print("==============")
#        continue    
        #question = input("Enter a question \n")
        #sys.stdout.flush()

        #if question == "quit":
        #    exit()
        
        #print("Question = " + question)
        #print("Chunks in cache = " + cache.size())
        
        #relevant_chunk = chunk_localization.predict(cache, question)
        
        #vqa_module = VQA(
        #    args.videoqa_config, 
        #    args.videoqa_model_path,
        #    args.videoqa_vocab_path,
        #    args.clip_num_c3d
        #)
        #vqa_module.predict(question, cache)
        #print("done")


main()
