// @ImagePlus(label="Training image", description="Stack or a single 2D image") image
// @ImagePlus(label="Label image", description="Image of same size as training image containing binary class labels") labels
// @ImagePlus(label="Test image", description="Stack or a single 2D image") testImage
// @Integer(label="Num. of samples", description="Number of training samples per class and slice",value=2000) nSamplesToUse
// @OUTPUT ImagePlus prob
import ij.IJ;
import trainableSegmentation.WekaSegmentation;
import hr.irb.fastRandomForest.FastRandomForest;
 
// starting time
 startTime = System.currentTimeMillis();
    
// create Weka segmentator
seg = new WekaSegmentation(image);
 
// Classifier
// In this case we use a Fast Random Forest
//rf = new FastRandomForest();
rf = new NaiveBayes();
// Number of trees in the forest
//rf.setNumTrees(200);
rf.setUseKernelEstimator(True);
          
// Number of features per tree
//rf.setNumFeatures(0);  
// Seed  
//rf.setSeed( (new java.util.Random()).nextInt() );    
// set classifier  
seg.setClassifier(rf);    
// Parameters   
// membrane patch size  
seg.setMembranePatchSize(11);  
// maximum filter radius
seg.setMaximumSigma(16.0f);
   
// Selected attributes (image features)
enableFeatures = new boolean[]{
            true,   /* Gaussian_blur */
            true,   /* Sobel_filter */
            true,   /* Hessian */
            true,   /* Difference_of_gaussians */
            true,   /* Membrane_projections */
            false,  /* Variance */
            false,  /* Mean */
            false,  /* Minimum */
            false,  /* Maximum */
            false,  /* Median */
            false,  /* Anisotropic_diffusion */
            false,  /* Bilateral */
            false,  /* Lipschitz */
            false,  /* Kuwahara */
            false,  /* Gabor */
            false,  /* Derivatives */
            false,  /* Laplacian */
            false,  /* Structure */
            false,  /* Entropy */
            false   /* Neighbors */
};
    
// Enable features in the segmentator
seg.setEnabledFeatures( enableFeatures );
    
// Add labeled samples in a balanced and random way
seg.addRandomBalancedBinaryData(image, labels, "class 2", "class 1", nSamplesToUse);
    
// Train classifier
seg.trainClassifier();
 
// Apply trained classifier to test image and get probabilities
prob = seg.applyClassifier( testImage, 0, true );
// Set output title
prob.setTitle( "Probability maps of " + testImage.getTitle() );
// Print elapsed time
estimatedTime = System.currentTimeMillis() - startTime;
IJ.log( "** Finished script in " + estimatedTime + " ms **" );