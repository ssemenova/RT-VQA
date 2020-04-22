# VQA

## Setup

### Setup TMLGA

1. Download requirements
  ~~~
  cd TMLGA
  sh ./downlad.sh
  ~~~
2. Download pretrained weights [here](https://zenodo.org/record/3590426/files/model_charades_sta) and move them to ``TMLGA/checkpoints/charades_sta``
3. To test that everything works, run

  ~~~
  python main.py --config-file=experiments/charades-sta.yaml
  ~~~


### Setup VideoQA

1. Download the [VGG16 checkpoint](https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM) and [C3D checkpoint](https://www.dropbox.com/sh/8wcjrcadx4r31ux/AAAkz3dQ706pPO8ZavrztRCca?dl=0) and put them in ``VideoQA/util``
2. Download the word embeddings trained over 6B tokens ([glove.6B.zip](https://nlp.stanford.edu/projects/glove/)) from GloVe, unzip them, and put the 300d file in directory ``VideoQA/util``
3. ``pip install -r PSAC/requirements.txt``

#### Pre-process Dataset
1. Download the [MSVD-QA dataset](https://mega.nz/#!QmxFwBTK!Cs7cByu_Qo42XJOsv0DjiEDMiEm8m69h60caDYnT_PQ) and place it in ``PSAC/MSVD-QA``
2. Download the youtube videos from the MSVD dataset ([YouTubeClips.tar in the downloads section](http://www.cs.utexas.edu/users/ml/clamp/videoDescription/)) and unzip them into ``PSAC/MSVD-QA/video``
3. Convert video names from garbled YouTubeClips video names into their corresponding video IDs, using [this file](https://mega.nz/#!QrowUADZ!oFfW_M5wAFsfuFDEJAIa2BeFVHYO0vxit3CMkHFOSfw) as the mapping between YouTubeClips.tar name and the real ID
4. Delete problematic videos with IDs: 
  - 451
  - 745
  - 1106
  (TODO: Figure out how to deal with these)
5. Pre-process the videos by running:
  ~~~
  python preprocess_msvdqa.py PSAC/MSVD-QA
  ~~~
