run("Image Sequence...", "open=C:/Users/maxim/Desktop/segmentation/data/testdata/ground_truth/1200.tif sort");
name_label=getImageID();
rename("label");
selectImage(name_label);


run("Image Sequence...", "open=C:/Users/maxim/Desktop/segmentation/data/testdata/original/1200.tif sort");
name_original=getImageID();
selectImage(name_original);

run("Run your network", "input=original normalizeinput=false percentilebottom=3.0 percentiletop=99.8 clip=false ntiles=4 blockmultiple=64 overlap=0 batchsize=8 modelfile=C:\\Users\\maxim\\Desktop\\segmentation\\models\\CSBDeep\\TF_SavedModel.zip showprogressdialog=true");

name_result=getImageID();
selectImage(name_result);
run("Duplicate...", "title=prediction duplicate");

name_prediction=getImageID();
selectImage(name_prediction);
rename("prediction");
setAutoThreshold("Default dark");
//run("Threshold...");
setOption("BlackBackground", true);
run("Convert to Mask", "method=Default background=Dark calculate black");
run("32-bit");
run("Enhance Contrast...", "saturated=0.3 normalize process_all");
selectImage(name_prediction);
run("Image Sequence... ", "format=TIFF name=[] use save=C:/Users/maxim/Desktop/segmentation/data/testdata/prediction/0000.tif");



function getIntensSum(window_name) 
{
selectWindow(window_name);
n_print=0;
sum_of_intens=0;
getStatistics(area, mean, min, max, std, histogram);
binWidth = (max-min)/256;
value = 2*binWidth;
for (i=1; i<=histogram.length-1; i++) 
{

   if (histogram[i]>0) {
       sum_of_intens+=value*histogram[i];
       n_print+=1;
   }
   value += binWidth;	
}
return sum_of_intens;
}


imageCalculator("Multiply create", "label","label");
rename("label_squared");
label_squared=getIntensSum("label_squared");
overlap=getIntensSum("label_squared");

imageCalculator("Multiply create", "prediction","prediction");
rename("prediction_squared");
prediction_squared=getIntensSum("prediction_squared");
overlap=getIntensSum("prediction_squared");

imageCalculator("Multiply create", "label","prediction");
rename("overlap");
overlap=getIntensSum("overlap");

dsc=2*overlap/(label_squared + prediction_squared);
print(dsc);
close("overlap");
close("label_squared");
close("prediction_squared");
