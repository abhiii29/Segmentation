//preprocessing
run("Reduce...", "reduction=2");
run("Invert", "stack");
run("8-bit");
setAutoThreshold("Default dark");
//run("Threshold...");
setOption("BlackBackground", true);
run("Convert to Mask", "method=Default background=Dark calculate black");
run("Enhance Contrast...", "saturated=0.3 normalize");


//batch processing
for (n=1; n<=nSlices; n++) 
{
	setSlice(n);
	run("Enhance Contrast...", "saturated=0.3 normalize");
	run("Distance Transform Watershed", "distances=[City-Block (1,2)] output=[16 bits] normalize dynamic=11 connectivity=4");
	setOption("ScaleConversions", true);
	run("8-bit");
	setAutoThreshold("Default dark");
	setThreshold(1, 255);
	setOption("BlackBackground", true);
	run("Convert to Mask");
	
	//enhancing the edges
	run("Options...", "iterations=2 count=1 black do=Erode");

	
	//removing any spots
	run("Fill Holes");

	//removing errors in edges
	run("Voronoi");
	run("Invert");
	
	//run("Brightness/Contrast...");
	run("Enhance Contrast", "saturated=0.35");
	run("Apply LUT");
	setAutoThreshold("Default dark");
	//run("Threshold...");
	setThreshold(255, 255);
	run("Convert to Mask");

	//filling up gaps
	run("Watershed Irregular Features", "erosion=11 convexity_threshold=0 separator_size=0-Infinity");

	//segmentation
	run("Distance Transform Watershed", "distances=[City-Block (1,2)] output=[16 bits] normalize dynamic=11 connectivity=4");

	//clearing boundaries
	makeOval(-20, 0, 2560, 2560);
	setBackgroundColor(255, 255, 255);
	run("Clear Outside", "slice");

	//saving the image
	saveAs("Tiff", "C:/Users/abga577c/Desktop/ProjectData/TempTestData/segmented/"+n+".tif");
	close();
	close();
}