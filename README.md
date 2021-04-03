# Parallel GPU Strategies Benchmark

Project from February - 2021

This repository contains code for the **Parallel GPU Strategies Benchmark** Project.

### Summary

This project aimed to propose a principled aproach to measure performance with different parallel strategies techniques.

### Storyline

I was working with a multi-gpu workstation and realised that despite of having 4,6,8+ gpus available, the overall performance didn't increased much. People around were aware of this issue but nobody could explain the causes or propose a definitive solution. So I worked around a bit and did some profiling of the training step. I found a huge computational overhead caused by some pre-defined device-to-device communication strategy. The NCCL All-reduce technique, as spoted below.

![prof1](https://github.com/patrick-schubert/parallel-strategies-benchmark-tensorflow2/blob/main/profilling-highlited.png)
![prof2](https://github.com/patrick-schubert/parallel-strategies-benchmark-tensorflow2/blob/main/profiling-top-durations.jpeg)

The NCCL is one of many other parallel strategies that could be used to make communications between devices. For Deep Learning training, is the best one given that for GPUs is better to reduce the batch in every single GPU than to communicate across devices to send their "reduced batch data", due to their extremelly fast vectorized computations.

(Deep Learning training time)

CPU: GPU, why don't you talk to me anymore?

GPU: I would rather sum than communicate :p

![parallelstrategies](https://github.com/patrick-schubert/parallel-strategies-benchmark-tensorflow2/blob/main/paralle-strategies.jpeg)

So I coded the benchmark and got some plots from a 4 and 8 GPUs machine (All Nvidia 2080ti GTX). The results were astonishing.

As the results reported, was better to send "batch data" to cpu, reduce there and send a copy to every gpu connected than to use a highlly optimized built-in parallel strategy. Without touching the machines, I could infer that there was no bus (NVLink) between gpus so latency was huge due to the unnecessary turn around that had to be made to make things work as pre-defined. 

This was an old problem from early 2018, and was affecting hugelly the productivity of the AI group, but then we got some promissing directions to improve overall performance of the group.

![bench1](https://github.com/patrick-schubert/parallel-strategies-benchmark-tensorflow2/blob/main/benchmark1.jpeg)
![bench2](https://github.com/patrick-schubert/parallel-strategies-benchmark-tensorflow2/blob/main/benchmark2.jpeg)




