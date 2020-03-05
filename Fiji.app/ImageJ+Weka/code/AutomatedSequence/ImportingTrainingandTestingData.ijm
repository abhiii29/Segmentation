run("Image Sequence...", "open=C:/Users/abga577c/Desktop/ProjectData/TempTestData/training/ground_truth/0100.tif sort");
run("Image Sequence...", "open=C:/Users/abga577c/Desktop/ProjectData/TempTestData/training/original/0100.tif sort");
run("Image Sequence...", "open=C:/Users/abga577c/Desktop/ProjectData/TempTestData/validation/original_val/0400.tif sort");
selectWindow("original_val");
rename("testdata");

