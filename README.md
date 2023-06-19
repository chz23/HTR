# Handwritten Text Recognition (ft. Object Detection)

<img src="https://github.com/chz23/HTR/assets/116154404/890058c5-35ab-4557-8d7b-f0b82e67c0bb" alt="Overview of Process" height="400">

---
## Why Do This?
---

> "I can practically mark this with my eyes closed!"
> 
> ~ Every educator, probably.

As a former educator under MOE, I'm ~~painfully~~ _keenly_ aware of the various pain points in the industry. Education, especially at the formative ages, typically requires a 'human' touch because children depend on us for more than just academic progress. That said, not _everything_ requires a human touch; furiously marking practice papers at 2am in the morning is a good example. 

---
## Overview / Vision
---

The end goal is __automation of the marking process__. 

Every project has a starting point; for this project, it is the __detection and recognition of handwritten text__. This means that the focus of this project is __computer vision__, specifically, __object detection and Handwritten Text Recognition (HTR)__, where HTR is essentially a more specialised branch of Optical Character Recognition (OCR). There will also be a final component, automated marking, which will utilise a simple matching process to __'mark' unambiguous, fill-in-the-blank worksheets__.

Based on a 70/20/10 split for train/validation/test, this worked out to:

| Dataset | No. of worksheets / words |
| --- | --- |
| Train | 172 worksheets / 1,775 words |
| Validation | 49 worksheets / 506 words |
| Test | 25 worksheets / 258 words |
| Total | 246 worksheets / 2,538 words |

__Note__: the above figures do _not_ include augmentations, ie., these are the 'raw' numbers.

---
## Part 1: Data Cleaning
---

<img src="https://github.com/chz23/HTR/assets/116154404/d42ade99-408d-403d-a79c-1f0d0917c086" alt="Expectation vs. Reality" height="400">

In a real-world setting, we assume that educators would be passing unmarked worksheets through the scanner. Unfortunately, as an ex-educator, all I have access to are hastily-taken pictures of marked spelling worksheets that I had forgotten to delete.

The process was as such:
1. Manually adjust edges of each worksheet, then apply CamScanner to obtain the 'scanned' look.
2. Load the images into an iPad, then manually erase marks made by the teacher (me...) using an Apple Pencil.
3. __Resize__ images.
4. __Draw bounding boxes__ around each handwritten portion (for the object detection model).
5. __Label__ each handwritten portion (for the HTR model).
6. Apply relevant __augmentations__.

__Note__: when gathering actual data from current educators / MOE, the first two steps can (fortunately) be omitted!

---
## Part 2: Object Detection (YOLOv8)
---

YOLOv8 purports to be both __efficient__ and __accurate__. It accomplishes the former by being a __single-stage detector__, and the latter through repeated improvements (across all YOLO versions); the latest, YOLOv8, has some useful additions such as upsampling, which increases resolution of feature maps to bypass the issue of having features with high semantic value but low resolution - basically, it's now __better at detecting small objects__, which is pretty important for handwritten text!

### Model Performance
---

| Metric | Score / Performance |
| --- | --- |
| mAP50 | 0.99 |
| mAP50-95 | 0.85 |

The metrics above are standard for object detection models these days, but they don't necessarily serve our purpose. __Object detection for the purposes of HTR has to be a lot more stringent.__ Here's an intuitive understanding of _why_, using IoU as an example:

<img src="https://github.com/chz23/HTR/assets/116154404/b641d28e-b4fb-4eb4-b69b-ee1233465fcb" alt="IoU" height="100">

Which image has a lower IoU? The one on the right.

Which has a ___useable___ result? The one on the __left__.

Basically, we're looking for bounding boxes that encompass the __entire word__. This isn't always obvious from metrics, unless the IoU is extremely close to 1.0.

Therefore, I opted to __manually inspect the test images__ instead. Here, I will define __'accurate'__ as the __predicted bounding box being able to capture the entire word__, ie., it is ready for passing into the HTR model. Therefore, 'inaccurate' will be everything else, ie., will cause issues when passed into the HTR model.

### Exploring Erroneous Predictions
---

Results: __254 out of 258 predicted bounding boxes were accurate__. This translates into 21 out of 25 worksheets (input images) being completely accurate; here is an example:

<img src="https://github.com/chz23/HTR/assets/116154404/ae992997-8cb4-4ed4-a847-6ee2c8daea3c" alt="042" height="500">

4 out of 25 worksheets had inaccurate predictions, with each worksheet having exactly 1 erroneous prediction; here is an example:

<img src="https://github.com/chz23/HTR/assets/116154404/37084d76-ee99-46e0-bb6e-763f1136f46d" alt="127" height="500">

In the image above, there is an issue with a false negative at question 4 - the model failed the pick up on the 'of'. This case is slightly special; rather than being a student's answer, it was a manual rectification of an error in the worksheet. In other words, there were very _few_ examples. In addition, unlike the other data points, this error was very close to the printed portions, and not written near/above a line. 

That said, this was the _only_ example of the model failing to pick it out. Therefore, an additional reason could be that this particular student's handwriting is somewhat similar to the printed portion, especially since it was a word with only 2 characters ('of').

### Model Evaluation
---

The model was able to __perform to a high level of accuracy__, which is relatively crucial because of the 'pipeline' nature of this project, where the output for this model serves as the input for the next model (HTR).

There were only 4 erroneous predictions, and they were generally different. In other words, the model did not experience a specific difficulty in a particular area; rather, it occasionally faced issues when the particular data point was dissimilar compared to the rest. This is understandable, since I worked with a comparatively small dataset. I believe __all of these issues can be overcome with a more robust and diversified dataset__ - just one set of worksheets per teacher country-wide would be plenty!

---
## Part 3: HTR (TrOCR)
---

TrOCR uses pre-trained image & text transformers, therefore bypassing CNNs entirely. Aside from being flexible & accurate, the language modelling aspect is not heavily emphasised upon, which is a ___good___ thing for this project. To clarify, some models offer an 'autocorrect' feature which can have its uses, but it's very unsuitable for students' work, because we want to recoognise the __exact__ word being written!

### Model Performance
---

| Trained on custom dataset? | Dataset | Word Error Rate (WER) | Accuracy (100% - WER) |
| --- | --- | --- | --- |
| No | Test | 36.0% | 64.0% |
| Yes | Test | 11.2% | 88.8% |
| Yes | Validation | 10.5% | 89.5% |
| Yes | Train | 1.7% | 92.3% |

__Note__: WER is a harsher metric than its character-level counterpart (Character Error Rate), but more applicable to this project as marking is done on the word, not character, level.

First, I verified whether there was even a need for my custom dataset by feeding my test data into the model _without_ training it on my own dataset. A WER of 36% (ie., word-level accuracy of 64%) is not bad, but certainly not good enough for our purposes. We're looking for a WER that would be acceptable by 'human' standards, ie., approximates the human error rate (which is unknown, but can roughly be estimated based on my own experience).

The results of training it on my own dataset are a lot more promising - a __test WER of 11.2% (ie., word-level accuracy of 88.8%)__. It doesn't quite rival actual humans yet, but this is certainly a good start, given that we've only explored one pretrained model and a small dataset.

### Exploring Erroneous Predictions
---

HTR is a lot more difficult to 'nail' than object detection. For the latter, the model only has one 'class' to contend with - it is essentially looking for the anomalies (handwritten portions) amidst a sea of printed text that never changes (ie., the background). 

For the former, the _style_ of writing is what matters; there are an infinite ways to write (insert letter of choice here). This is when the size of the dataset became an issue, something which became apparent while looking into erroneous predictions. The worst offender was messy + inconsistent handwriting; both worksheets shown below are by the same student:

<img src="https://github.com/chz23/HTR/assets/116154404/9f3bb8e6-6a38-4048-a422-3b711d855034" alt="HTR error" height="300">

Some other errors included the following:

<img src="https://github.com/chz23/HTR/assets/116154404/9d34d9a2-f365-4dd1-bff1-5cd38a23c247" alt="HTR error (others)" height="250">

This harks back to when I mentioned the language modelling aspect being present but not overly so, as can be seen in the contradicting errors made in the first and last errors. The 2nd error is understandable - lots of people would mistake the 't' as an 'x' (in fact, perhaps it _should_ be counted as an 'x'...), while the 3rd error is slightly puzzling - probably goes back to the issue of the dataset size.

### Model Evaluation
---

The results are promising despite the small dataset (2,538 words across 33 different sets of handwriting (or more; some students enjoy 'experimenting'...)), which bodes well for real-life deployment of the model. 

---
## Moving Forward
---

Speaking from experience, if the marking process could be automated, a good __30% to 50% of time could be freed up__, which educators could re-allocate towards other duties, such as professional development or helping students to bridge learning gaps, thus resulting in a positive cycle where students are able to reap more benefits from their education, and educators feel more deeply satisfied with their contributions. It seems to be a large-scale, ambitious project, but that isn't necessarily the case.

This humble, small-scale project has shown that __detecting and recognising handwriting is indeed very plausible, even with a considerably small dataset__. With the support of an organisation such as MOE, it would definitely be possible to deploy it on a large scale for a multitude of purposes from digitisation to automation of marking. The latter itself encompasses many areas from simple marking (matching of 'decoded' answers to answer key) to ambitious NLP projects (essay marking) that would likely require the latest technology and beyond. While automation of essay-marking indeed will require a lot of time and resources, it's just one portion of marking. Just being able to automate marking for simple fill-in-the-blank worksheets would be a major time-saver. It was a pipe dream (that we endlessly joked about) in the recent past when I was still an educator, but no longer!
