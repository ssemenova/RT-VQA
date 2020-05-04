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


video_count = 0
cache = None

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
    # If using 'video_dir', optionally pass in test_length, which will
    # dictate how many videos to play during the test. Default is set
    # to 502 because that's how many videos are in the MSVD-QA test
    # dataset.
    parser.add_argument('--test_length', type=int, default='502')
    # If using 'video_dir', optionally pass in the option to filter 
    # out large videos. Not good for real testing, but useful for 
    # if you need to play the videos through X11.
    parser.add_argument(
        '--filter_large_videos', action='store_true'
    )
    # Whether to display the video or not
    parser.add_argument(
        '--display_video', action='store_true'
    )
    ## CACHE VARIABLES ##
    # Cache size (in terms of chunks)
    parser.add_argument('--cache_size', type=int, default='10')
    # How often to evict (in terms of chunks). A lower number 
    # (more frequent evictions) means the chunk creation and 
    # cache insertion process takes longer.
    parser.add_argument('--evict_mod', type=int, default='10')
    # Whether to use RAM for the cache
    parser.add_argument('--use_ram', action='store_true')

    ## VIDEO PROCESSING VARIABLES ##
    # The size of a video chunk to cache
    parser.add_argument('--chunk_size', type=int, default='50')

    ## C3D FEATURE EXTRACTION ##
    # Depending on how these variables are set, some video data
    # might be lost. A chunk is divided into evenly-spaced clips
    # of some fixed size. If clip frames don't overlap then video
    # data is lost. Likewise, if clips frames overlap too much,
    # potentially too much computation occurs with no benefit.

    # Amount of clips to create per chunk. Can be <= chunk_size
    parser.add_argument('--clip_num_c3d', type=int, default='5')
    # Frames per clip. Can be <= chunk_size
    parser.add_argument('--frames_per_clip_c3d', type=int, default='5')

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


def _create_test(
        video_directory, test_length, filter_large_videos
    ):
    print("Creating test")
    interleaving_temp = []
    test_qa = pd.read_json('VideoQA/MSVD-QA/test_qa.json')

    for f in os.listdir(video_directory):
        video_id = f.strip('YouTubeClipsvid.a')
        if video_id:
            interleaving_temp.append(video_id)
   
    random.seed()
    random.shuffle(interleaving_temp)
    questions = []
    interleaving = []
    
    for i in range(test_length):
        video_id = interleaving_temp[i]

        relevant_qs = test_qa[test_qa.video_id == int(video_id)]
        relevant_qs_len = len(relevant_qs)

        if relevant_qs_len != 0:
            # The chosen amount of questions should depend on the 
            # video length ... larger videos get more questions
            v = _open_video(
                video_directory + 'YouTubeClipsvid' + video_id + '.avi'
            )
            v.set(cv2.CAP_PROP_POS_AVI_RATIO,1)
            video_len = v.get(cv2.CAP_PROP_POS_MSEC)
            
            video_width = v.get(cv2.CAP_PROP_FRAME_WIDTH)
            if filter_large_videos and video_width < 800: 
                sample_max = int(min(relevant_qs_len, video_len / 3))
                sample_size = random.randint(
                    1, sample_max
                )
                sampled_q_ids = random.sample(
                    range(relevant_qs_len),
                    k=sample_size
                )
                sampled_qs = []
    
                for question in sampled_q_ids:
                    sampled_qs.append(
                        [relevant_qs.iloc[question].question,
                        relevant_qs.iloc[question].answer]
                    )
    
                questions.append(sampled_qs)
                interleaving.append(
                    'YouTubeClipsvid' + video_id + '.avi'
                )

    return interleaving, questions


def _open_video(video_name):
    cap = cv2.VideoCapture(video_name)
    while not cap.isOpened():
        cap = cv2.VideoCapture(video_name)
        cv2.waitKey(1000)
    return cap


def _get_features(chunk_id):
    global cache
    cache.commit(chunk_id)
    logging.debug(cache.newest_id)

def process_video(
    chunk_size, 
    frames_per_clip_c3d, clip_num_c3d,
    i3d_extractor_model_path,
    video_name,
    video_dir,
    interleaving,
    display_video,
    args, questions # TODO - maybe don't pass these in here
):
    global video_count
    global cache

    # TODO: LATER-- replace this with something more real-time
    # and not frame-by-frame
    if interleaving:
        cap = _open_video(video_dir + "/" + video_name[0])
    else:
        cap = _open_video(video_name)

    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    
    frame_count = 0
    chunk_count = 1

    cache.new_chunk(
        chunk_size,
        frames_per_clip_c3d, clip_num_c3d,
        i3d_extractor_model_path
    )

    ask_questions_thread = threading.Thread(
        target=ask_questions, args=(
            args, questions)
    )
    ask_questions_thread.start()

    # Force roughly 30 FPS
    starttime=time.time()

    print("Playing videos...")
    while True:
        flag, frame = cap.read()
        if flag:
            pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
           
            if frame_count % 2 == 0 and display_video:
                cv2.imshow('frame', frame)

            logging.debug("processing frame #" + str(pos_frame))
            if frame_count == chunk_size:
                logging.debug(
                    "creating chunk #" + str(cache.current_chunk.id
                ))
                
                get_features_thread = threading.Thread(
                  target=_get_features, args=(
                    [chunk_count])
                )
                get_features_thread.start()
               
                frame_count = 0
                print(cache.newest_id)
                cache.new_chunk(
                    chunk_size,
                    frames_per_clip_c3d,
                    clip_num_c3d,
                    i3d_extractor_model_path
                )

                chunk_count += 1
                time.sleep(100000)
            else:
                cache.db.get(chunk_count).add_frame(
                    frame, frame_count
                )

            frame_count += 1
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
            print("frame is not ready")
            cv2.waitKey(1000)

        time.sleep(.25 - ((time.time() - starttime) % .25))
        if cv2.waitKey(10) == 27:
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            cap.release()
            video_count += 1
            if video_count < len(video_name):
                cap = _open_video(video_dir + "/" + video_name[video_count])
            else:
                break


# Notcurrently used.
def kill_old_threads(cache):
    while True:
        ids_to_kill = [(k, v) for k, v in running_threads.items() if k < cache.oldest_id]
        for id in ids_to_kill:
            running_threads.get(id).stop()

        time.sleep(10)


def ask_questions(args, questions):
    global cache

    question_count = 0
    time.sleep(10)
   
    vqa_module = VQA(
        args.videoqa_config,
        args.videoqa_model_path,
        args.videoqa_vocab_path,
        args.clip_num_c3d
    )
    
    while True:
        # question = input("Enter a question \n")
        # sys.stdout.flush()

        # if question == "quit":
        #    exit()

        # print("Question = " + question)
        # print("Chunks in cache = " + cache.size())
        
        # Ask questions about the past 5 videos for now
        random.seed()
        random_video_index = random.randint(
            max(video_count - 5, 0), 
            max(video_count - 1, 0)
        )
        if len(questions[random_video_index]) != 0:
            random_question_index = random.randint(
                0,
                len(questions[random_video_index]) - 1
            )
            question = questions[random_video_index][random_question_index]

            questions[random_video_index].remove(question)

        # relevant_chunk = chunk_localization.predict(cache, question)
        
            print("Asking question = " + str(question[0]))
            import pdb; pdb.set_trace()
            answer = vqa_module.predict(question[0], cache)
        

            question_count += 1
            time.sleep(5)


def main():
    global cache

    logging.basicConfig(filename='VQA.log',level=logging.DEBUG)
    args = parse_args()

    cache = Cache(
        args.database_name,
        args.cache_size,
        args.evict_mod,
        args.use_ram,
    )

    interleaving, questions = _create_test(
        args.video_dir,
        args.test_length,
        args.filter_large_videos
    )

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
    
    process_video(
        args.chunk_size,
        args.frames_per_clip_c3d,
        args.clip_num_c3d,
        args.i3d_extractor_model_path,
        interleaving,
        args.video_dir,
        True,
        args.display_video,
        args, questions
    )


main()
