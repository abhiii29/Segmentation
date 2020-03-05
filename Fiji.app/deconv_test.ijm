run("Image Sequence...", "open=C:/Users/maxim/Desktop/segmentation/data/testdata/original/1200.tif sort");
run("Deconvolution - Microtubules", "input=original ntiles=1 batchsize=1");
selectWindow("result");
run("Duplicate...", " ");
saveAs("Tiff", "C:/Users/maxim/Desktop/segmentation/data/testdata/prediction/result-1.tif");