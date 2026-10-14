# test_classification_loader.cpp

ClassificationLoader on PNGs in class folders, and parallel_for.

## DiscoversClassesAndYieldsNchwOneHotBatches

Five images in the cat and dog folders yield an NCHW batch [2, 3, 8, 8] and one-hot rows that sum to 1.

## CoversEverySampleAcrossPrefetchedBatchesWithoutOverlap

With batch size 2, five images come out in three batches (2+2+1), and another get_batch after the end throws.

## ResetRestartsEpochAndAllowsReuse

After the epoch is exhausted, reset() allows reading from the start again, and the first batch has 4 images.

## LockedClassNamesKeepOneHotWidthWhenAFolderIsMissing

The class vocabulary from train (cat, dog) keeps the one-hot width at 2, and the dog folder alone receives [0, 1].

## RejectsInvalidConstructorArguments

Batch 0, image side 0, and a missing directory are rejected.

## ExecutesEveryIndexExactlyOnce

Each of 64 indices runs once, and the thread count for 128 tasks stays within the cap of 16.

## PropagatesWorkerExceptions

An exception from one index leaves parallel_for after the workers are joined.
