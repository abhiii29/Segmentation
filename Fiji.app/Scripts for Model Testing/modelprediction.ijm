for(i=1;i<2;i++)
{
//run("Image Sequence...", "open=C:/Users/maxim/Desktop/segmentation/data/testdata/original/1200.tif sort");
open("C:/Users/maxim/Desktop/segmentation/data/testdata/original/1200.tif");
name_original=getImageID();
//selectImage(name_original);
run("Run your network", "input=original normalizeinput=false percentilebottom=3.0 percentiletop=99.8 clip=false ntiles=4 blockmultiple=64 overlap=0 batchsize=8 modelfile=C:\\Users\\maxim\\Desktop\\segmentation\\models\\CSBDeep\\TF_SavedModel.zip showprogressdialog=true");
selectImage(name_original);
close();
}
print(result);
//selectWindow("result");
//name_result=getImageID();
//selectImage(name_result);
//saveAs("Tiff", "C:/Users/maxim/Desktop/segmentation/data/testdata/prediction/result.tif");
//close();
//run("Image Sequence... ", "format=TIFF name=[] use save=C:/Users/maxim/Desktop/segmentation/data/testdata/prediction/0000.tif");
/*
name_result=getImageID();
selectImage(name_result);
run("Duplicate...", "duplicate");
rename("prediction");
setAutoThreshold("Default dark");
//run("Threshold...");
setOption("BlackBackground", true);
run("Convert to Mask", "method=Default background=Dark calculate black");
run("32-bit");
run("Enhance Contrast...", "saturated=0.3 normalize process_all");
*/
