# Handwritten Text Recognition (ft. Object Detection)

<img src="https://github.com/chz23/HTR/assets/116154404/a7051ae3-4215-4a2a-b58b-ea012bc53424" alt="t2w5_original_edited" height="400">

---
## Why Do This?
---

> "I can practically mark this with my eyes closed!"
> - Every educator, probably.

As a former educator under MOE, I'm ~~painfully~~ _keenly_ aware of the various pain points in the industry. Education, especially at the formative ages, typically requires a 'human' touch because children depend on us for more than just academic progress. That said, not _everything_ requires a human touch. Furiously marking practice papers at 2am in the morning while being dismayed at the contrast in the effort being put in by the students vs. myself is a good example. An AI can't design & deliver the student-specific lectures required to jolt said students into action, but it can certainly take over the tedious, repetitive marking process. 

Digitalisation has impacted our lives in many ways. As we make continuous progress in this area, there is one sector that is struggling to keep up with the pace: the education sector, especially at the primary school level, where the traditional pen-and-paper method is still extremely widely-used. There exists some resistance against the movement in general - a common complaint is that the disadvantages (especially time) outweigh the advantages of using digital tools. In addition, less tech-savvy educators find it to be a challenging and frustrating endeavour overall.

These are valid complaints that we can work on with the help of machine learning models. The key here is to change the experience from a negative to a positive one by coming up with a model that is both easy to use and has the crucial benefit of saving educators precious time. Educators often joke about wanting to buy 'marking machines' because of the sheer amount of time spent on marking students' work, an important but time-consuming process that seemingly cannot be improved on any further. Using machine learning, we can finally tackle this problem.

To this end, I will build a Handwritten Text Recognition (HTR) model to digitalise handwritten text using CRNN / seq2seq models. In addition to the outcome of digitalising handwritten text, I will include an example of how it can be implemented by automating certain marking processes. Chiefly, this project aims to provide a simple, practical and tangible starting point to help educators recognise the value of embracing the digital age, which will in turn pave the way for even more progress to be made in this field. 

Model (HTR) performance will be guided by Character Error Rate (CER) as well as Word Error Rate (WER).
