filepath=File.openDialog("SELECT A TEST PREDICTION FILE");
imageDir=File.directory;
fileList = getFileList(imageDir); 
numberSlice=fileList.length;
run("Image Sequence...", 
  "open=[&filepath]"+
  " number="+numberSlice+
  " starting=1"+
  " increment=1"+
  " scale=100 "+
  "file=[.tif] "+
  "sort");
rename("prediction");


filepath=File.openDialog("SELECT A TEST LABEL FILE");
imageDir=File.directory;
fileList = getFileList(imageDir); 
numberSlice=fileList.length;
run("Image Sequence...", 
  "open=[&filepath]"+
  " number="+numberSlice+
  " starting=1"+
  " increment=1"+
  " scale=100 "+
  "file=[.tif] "+
  "sort");
rename("label");


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

