names=Array.getSequence(n);
setAutoThreshold(""+names[best]+" dark");
//run("Threshold...");
//setThreshold(129, 255);
setOption("BlackBackground", true);
run("Convert to Mask");