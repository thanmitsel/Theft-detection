function [C, rbf_sigma,minobjfn] = tuneSVM(Data, Label, varargin)
% Function to tune SVM parameters C (box constraint) and sigma(for RBF
% kernel) using search methods. Data refers to the input to SVM (features). Provide only the
% training data in this argument. Keep mutually exclusive holdout testing
% set to obtain out-of-sample classification accuracy using the tuned SVM parameters.
% Supply the corresponding label in the argument 'Label'. 
% 
% Required inputs:
% Data - Training data where rows correspond to instances/observations (numobs = m) and
% columns correspond to features (numfeatures = n)
% Label - Training label corresponding to Data. Size should be (m,1).
%
% Optional parameters:
% Provide all optional parameters as name-value pair arguments.
% 'kernel' - Type of kernel function used in SVM. Expected arguments are
% 'linear' and 'rbf'. Default: 'rbf'. 
%
% 'method' - search method to be used. Default is 'neldearmead'
% minimization. Possible values are : 'neldermead','genetic' (genetic algorithm search),
% 'lhs'(Latin hypercube search),'GPSPosBasis2N','GPSPosBasisNp1','GSSPosBasis2N','GSSPosBasisNp1','MADSPosBasis2N',
% 'MADSPosBasisNp1'
%
% 'numFolds' - number of folds in the cross validation partition object
% created using "cvpartition". Default : 10. 
%
% 'boxconstraint' - parameter boxconstraint or C of SVM which is a
% trade-off between number of misclassified instances and separability of
% classes. Default: 1
%
% 'sigma' - parameter sigma of RBF kernel. Default : 1.
%
% 'searchpoint' - specify the search point (C,sigma) for minimizing the
% objective function. Expected values : 'random', 'specified'. Default:
% 'random'. 'specified' - the specified C,sigma values will be used to
% start the search. 'random' - random points will be used for start of
% search. 
%
% 'LowerBound'- LowerBound value for constrained optimization of SVM
% parameters. Default: -5.
%
%'UpperBound' - UpperBound value for constrained optimization. Default: 5
%
% 'TolMesh' - tolerance value of mesh size. Default: 1e-3. 
% 
% 'UseParallel' - when set to 'true', the algorithm will use parallel
% processing. Default: 'false'
% 
% Outputs: C - crossvalidated, tuned C value
% sigma - crossvalidated, tune sigma value for rbf kernel
% minobjfn - value of objective, minimization function at crossvalidated C
% or C,sigma
%
% Notes: 1. This function uses 'patternsearch' function for stable
% optimizations which requires MATLAB Global Optimization Toolbox. 
% 2. When you set the method to 'neldearmead', MATLAB uses the 'fminsearch'
% built-in function. This procedure does not guarantee convergence to
% global minima if the surface is inundated with local minima. In this
% case, repeat the search 20times with different search start points. Use
% the values for C and sigma with the least minobjfn. 

% Parse input arguments
p = inputParser; 
% Add defaults for optional inputs
addRequired(p,'Data',@ismatrix)
addRequired(p,'Label',@ismatrix)
expkern = {'linear';'rbf'};
addParameter(p,'kernel','rbf', @(x) any(validatestring(x,expkern)));
dmethod = 'neldermead'; % default method
expectedmethods = {'neldermead';'genetic';'lhs';'GPSPosBasis2N';...
    'GPSPosBasisNp1';'GSSPosBasis2N';'GSSPosBasisNp1';'MADSPosBasis2N';'MADSPosBasisNp1'};
addParameter(p,'method',dmethod, @(x) any(validatestring(x,expectedmethods)));
addParameter(p,'numFolds',10,@isnumeric)
addParameter(p,'boxconstraint',1,@isnumeric);
addParameter(p,'sigma',1,@isnumeric);
expectedsearchpts = {'random';'specified'};
addParameter(p,'searchpoint','random',@(x) any(validatestring(x,expectedsearchpts)));
addParameter(p,'LowerBound',-5,@isnumeric)
addParameter(p,'UpperBound',5,@isnumeric)
addParameter(p,'TolMesh',1e-3,@isnumeric)
addParameter(p,'UseParallel',false, @islogical)
parse(p,Data,Label,varargin{:})
Data = p.Results.Data; Label = p.Results.Label;
kernel = p.Results.kernel;
numFold = p.Results.numFolds;
method = p.Results.method; boxcon = p.Results.boxconstraint;
sigma = p.Results.sigma; LB = p.Results.LowerBound;
UB = p.Results.UpperBound; TolMesh = p.Results.TolMesh; searchpts = p.Results.searchpoint;
parall = p.Results.UseParallel;
clear('p')

% Create cross validation partitions
c= cvpartition(Label,'KFold',numFold);
% Determine starting point of search 
switch searchpts
    case 'random'
        z = randn(2,1);
    case 'specified'
        z = [boxcon; sigma]; % user specified C and sigma
end
% Obtain objective function to minimize based on kernel
switch kernel
    case 'linear'
        minfn = @(z)kfoldLoss(fitcecoc(Data,Label,'Learners',templateSVM('Standardize',1,'BoxConstraint',...
            exp(z(1)),'KernelFunction','linear'),'Crossval','on','CVPartition',c));
    case 'rbf'
        minfn = @(z)kfoldLoss(fitcecoc(Data,Label,'Learners',templateSVM('Standardize',1,'BoxConstraint',...
            exp(z(1)),'KernelScale',exp(z(2)),'KernelFunction','rbf'),'Crossval','on','CVPartition',c));
end
% Obtain search method
switch method
    case 'neldermead'
        searchm = @searchneldermead;
    case 'genetic'
        searchm = @searchga;
    case 'lhs'
        searchm = @searchlhs;
    case 'GPSPosBasis2N'
        searchm = @searchGPSPositiveBasis2N;
    case 'GPSPosBasisNp1'
        searchm = @searchGPSPositiveBasisNp1;
    case 'GSSPosBasis2N'
        searchm = @searchGSSPositiveBasis2N;
    case 'GSSPosBasisNp1'
        searchm = @searchGSSPositiveBasisNp1;
    case 'MADSPosBasis2N'
        searchm = @searchMADSPositiveBasis2N;
    case 'MADSPosBasisNp1'
        searchm = @searchMADSPositiveBasisNp1;
end
opts = psoptimset ('TolMesh',TolMesh,'SearchMethod',searchm,'UseParallel',parall);
[searchmin, minobjfn] = patternsearch(minfn, z,[],[],[],[],LB,UB,opts);
C = exp(searchmin(1,1)); rbf_sigma = exp(searchmin(2,1));
end
