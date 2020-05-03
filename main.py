import argparse
import cv2
import time
import threading
import sys
import logging
import random
import os
import pandas as pd

from chunk import Chunk
from cache import Cache
from chunk_localization import Chunk_Localization
from vqa import VQA
from other import StoppableThread


running_threads = {}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--database_name', type=str, default='vqadb')

    ## VIDEO MODE ##
    # Enter either 'video_dir' or 'video_name'.
    # Using a video_dir will run a random interleaving of videos in
    # the directory, using a video_name will just run the video.
    parser.add_argument(
        '--video_dir', type=str,  default='VideoQA/MSVD-QA/video/'
    )
    parser.add_argument('--video_name', type=str, default='./out.mp4')

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
    # Currently not being used.
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


def _create_test(video_directory):
    interleaving_temp = []
    test_qa = pd.read_json('VideoQA/MSVD-QA/test_qa.json')

    for f in os.listdir(video_directory):
        video_id = f.strip('YouTubeClipsvid.a')
        print(video_id)
        if video_id:
            interleaving_temp.append(video_id)
    
    random.shuffle(interleaving_temp)

    questions = []
    interleaving = []
   
    import pdb; pdb.set_trace()
    for video_id in interleaving_temp:
        relevant_qs = test_qa[test_qa.video_id == int(video_id)]
        relevant_qs_len = len(relevant_qs)
        if relevant_qs_len != 0:
            sample_size = random.randint(
                0, relevant_qs_len
            )
            sampled_q_ids = random.sample(
                range(relevant_qs_len),
                k=sample_size
            )
            sampled_qs = []
    
            for question in sampled_q_ids:
                sampled_qs.append(
                    relevant_qs.iloc[question].question
                )
    
            questions.append(sampled_qs)
            interleaving_temp.append(
                'YouTubeClipsvid' + video_id + '.avi'
            )
 
    return interleaving, questions


def _open_video(video_name):
    cap = cv2.VideoCapture(video_name)
    while not cap.isOpened():
        cap = cv2.VideoCapture(video_name)
        cv2.waitKey(1000)
    return cap


def _commit_current_chunk(chunk):
    return chunk.commit()


def process_video(
    chunk_size, cache, 
    frames_per_clip_c3d, clip_num_c3d,
    i3d_extractor_model_path,
    video_name,
    video_dir,
    interleaving
):
    # TODO: LATER-- replace this with something more real-time
    # and not frame-by-frame
    if interleaving:
        cap = _open_video(video_name)
    else:
        cap = _open_video(video_dir + "/" + video_name[0])

    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    
    chunk_count = 0
    frame_count = 0
    video_count = 0
    current_chunk = Chunk(
        cache, chunk_count, chunk_size,
        frames_per_clip_c3d, clip_num_c3d,
        i3d_extractor_model_path
    )

    while True:
        flag, frame = cap.read()
        if flag:
            pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            
            logging.debug("processing frame #" + str(pos_frame))
            if frame_count == chunk_size:
                logging.info("creating chunk #" + str(chunk_count))
                create_chunk_thread = threading.Thread(
                  target=_commit_current_chunk, args=(
                      current_chunk,
                    )
                )

                running_threads.update({
                    chunk_count: create_chunk_thread
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


def main():
    logging.basicConfig(filename='VQA.log',level=logging.DEBUG)
    args = parse_args()
    
    cache = Cache(
        args.database_name,
        args.cache_size,
        args.evict_mod,
        args.use_ram,
    )

    interleaving, questions = _create_test(args.video_dir)

    import pdb; pdb.set_trace()
    # Not currently used.
    #chunk_localization = Chunk_Localization(
    #    args.config_file_path,
    #    args.vocab_file_path,
    #    args.embeddings_file_path,
    #    args.max_question_length,
    #    args.min_question_length,
    #    args.chunk_size
    #)

    print("Setup done.")
    print("==============")
    
    process_video_thread = threading.Thread(
     target=process_video, args=(
            args.chunk_size,
            cache,
            args.frames_per_clip_c3d,
            args.clip_num_c3d,
            args.i3d_extractor_model_path,
            interleaving,
            args.video_dir,
            True
       )
    )
    process_video_thread.daemon = True

    kill_old_threads_thread = threading.Thread(
        target=kill_old_threads, args=(cache,)
    )

    process_video_thread.start()
    kill_old_threads_thread.start()
    
    question_count = 0
    while True:
#        print("==============")
#        continue    
        # question = input("Enter a question \n")
        # sys.stdout.flush()

        # if question == "quit":
        #    exit()
        
        # print("Question = " + question)
        # print("Chunks in cache = " + cache.size())
        q = questions[question_count]

        # relevant_chunk = chunk_localization.predict(cache, question)
        
        vqa_module = VQA(
           args.videoqa_config, 
           args.videoqa_model_path,
           args.videoqa_vocab_path,
           args.clip_num_c3d
        )
        vqa_module.predict(q, cache)

        question_count += 1

        # print("done")


main()
