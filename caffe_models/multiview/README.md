`multires_results.mat` contains combined fc7 features from multi-resolution inputs (standard rendering, sphere-30 rendering and sphere-60 rendering) and trained SVM model (using liblinear) for 40-class classification on these models.

In the mat file:
`combined_train_fc7` is a 9843x12288 matrix where `combined_train_fc7(:,1:4096)` are multiview CNN features from standard renderings, `combined_train_fc7(:,4097:8192)` are from sphere-30 renderings and `combined_train_fc7(:,8193:12288)` are from sphere-60 renderings.

`combined_train_label` is a 9843x1 vector containing class labels for the ModelNet40 train set shapes, its from 1 to 40 corresponding to alphabetic order of shape names (airplane is of label 1, xbox is of label 40)

`train_shape_ids` is a cell-array of shape IDs (such as `airplane_0001`), `combined_train_fc7` is organized in the same order as those IDs.

it's similar for the test set.

`model` is a liblinear trained SVM model we used. it's acquired by `model = train(combined_train_label, sparse(combined_train_fc7), '-s 1 -q -c 0.00001');
` where the cost value is choosed by cross-validation.

By using `model` on combined fc7 features one can achieve 93.8% average instance accuracy and 91.4% average class accuarcy on ModelNet40 test set.
