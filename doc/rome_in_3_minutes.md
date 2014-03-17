# Reconstructing Rome in Three Minutes


**Abstract**

A 3D model of the city of Rome is reconstructed from 500hrs of depth streams captured with 10,000 Kinects on a 1000-GPU-server in three minutes (234 frames per second per GPU)(1).

The core contribution is a new, flattened tree-data structure that supports all tree operations without the space- and time overhead introduced by child-pointers.

To prove the efficiency and versatility of the data structure, implementations on mobile devices and desktops are provided which are capable of reconstructing 3D models in realtime at the same quality as the GPU-version, except on a smaller scale.

*Footnotes*
1. The data set is 24Terabytes large. Hard-drive access is not included in the timings.


## Introduction

The field of 3D reconstruction is well explored [possibly give a definition/define the scope of '3D reconstruction'].[Research related work and cite a bit, demonstrating how problems like tracking, reconstruction, loop closure, etc. have been successfully tackled/solved by various methods].

The field of *realtime* 3D reconstruction is less mature, even less so on mobile devices that lack the computing power of top-notch [come up with better word] GPUs used in most systems/experiments. [Cite related work and highlight some of their short-comings each, proving the point made]. This work aims to allow everybody to create beautiful 3D reconstructions with commodity hardware. This [dont repeat yourself] is achieved via a new, flattened tree data structure that supports all tree operatoins without the space- and time overhead introduced by child-pointers. The flat layout leads to low data access divergence allowing efficient implementations on GPUs and modern processors [comment about modern processor cache characteristics]. It supports reconstructing at the same quality across a variety of hardware (from mobile phones to servers) at a suitable/appropriate [geeignet] scale. Its features include loop closure, …[comprehensive (= long) list of features] [Have to sound very excited and get the user excited for a smooth transition to the closing sentence of the introduction]

Nero burned Rome in three days. Let's rebuild it in three minutes.
