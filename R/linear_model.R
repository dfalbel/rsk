# LinearRegression --------------------------------------------------------

#' Linear Regression
#'
#' Ordinary least squares Linear Regression.
#'
#' @param x matrix. Training Data
#' @param y matrix. Target Values
#' @param fit_intercept boolean, optional whether to calculate the intercept for
#' this model. If set to false, no intercept will be used in calculations (e.g.
#' data is expected to be already centered).
#' @param normalize boolean, optional, default False. If True, the regressors X
#' will be normalized before regression. This parameter is ignored when
#' fit_intercept is set to False. When the regressors are normalized, note that
#' this makes the hyperparameters learnt more robust and almost independent of
#' the number of samples. The same property is not valid for standardized data.
#' However, if you wish to standardize, please use preprocessing.StandardScaler
#' before calling fit on an estimator with normalize=False.
#' @param copy_X boolean, optional, default True. If True, X will be copied;
#' else, it may be overwritten.
#' @param n_jobs int, optional, default 1. The number of jobs to use for the
#' computation. If -1 all CPUs are used. This will only provide speedup for
#' n_targets > 1 and sufficient large problems.
#'
#' @name LinearRegression
NULL

#' @rdname LinearRegression
rsk_LinearRegression <- R6::R6Class(
  "rsk_LinearRegression",
  inherit = rsk_model,
  public = list(
    initialize = function(fit_intercept, normalize, copy_X, n_jobs){
      private$pointer <- sklearn$linear_model$LinearRegression(fit_intercept, normalize, copy_X, n_jobs)
      private$pickle <- pickle$dumps(private$pointer)
    }
  )
)

#' @rdname LinearRegression
#' @export
LinearRegression <- function(x, y, fit_intercept = TRUE, normalize = FALSE,
                             copy_X = TRUE, n_jobs = 1){
  model <- rsk_LinearRegression$new(fit_intercept, normalize, copy_X, n_jobs)
  model$fit(x, y)
  return(model)
}

predict.rsk_LinearRegression <- function(model, x, ...){
  model$predict(x)
}

# LogisticRegression ------------------------------------------------------

#' Logistic Regression
#'
#' Logistic Regression (aka logit, MaxEnt) classifier.
#' In the multiclass case, the training algorithm uses the one-vs-rest (OvR) scheme
#' if the ‘multi_class’ option is set to ‘ovr’, and uses the cross- entropy loss if
#' the ‘multi_class’ option is set to ‘multinomial’. (Currently the ‘multinomial’
#' option is supported only by the ‘lbfgs’, ‘sag’ and ‘newton-cg’ solvers.)
#' This class implements regularized logistic regression using the ‘liblinear’
#' library, ‘newton-cg’, ‘sag’ and ‘lbfgs’ solvers. It can handle both dense and
#' sparse input. Use C-ordered arrays or CSR matrices containing 64-bit floats
#' for optimal performance; any other input format will be converted (and copied).
#' The ‘newton-cg’, ‘sag’, and ‘lbfgs’ solvers support only L2 regularization
#' with primal formulation. The ‘liblinear’ solver supports both L1 and L2
#' regularization, with a dual formulation only for the L2 penalty.
#' Read more in the [User Guide](http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression).
#'
#' @param x matrix. Training Data
#' @param y matrix. Target Values
#' @param penalty str, 'l1' or 'l2', default: 'l2'
#' Used to specify the norm used in the penalization. The 'newton-cg',
#' 'sag' and 'lbfgs' solvers support only l2 penalties.
#' @param dual bool, default: False
#' Dual or primal formulation. Dual formulation is only implemented for
#' l2 penalty with liblinear solver. Prefer dual=False when
#' n_samples > n_features.
#' @param C float, default: 1.0
#' Inverse of regularization strength; must be a positive float.
#' Like in support vector machines, smaller values specify stronger
#' regularization.
#' @param fit_intercept bool, default: True
#' Specifies if a constant (a.k.a. bias or intercept) should be
#' added to the decision function.
#' @param intercept_scaling float, default 1.
#' Useful only when the solver 'liblinear' is used
#' and self.fit_intercept is set to True. In this case, x becomes
#' (x, self.intercept_scaling),
#' i.e. a "synthetic" feature with constant value equal to
#' intercept_scaling is appended to the instance vector.
#' The intercept becomes ``intercept_scaling * synthetic_feature_weight``.
#  Note! the synthetic feature weight is subject to l1/l2 regularization
#' as all other features.
#' To lessen the effect of regularization on synthetic feature weight
#' (and therefore on the intercept) intercept_scaling has to be increased.
#' @param class_weight dict or 'balanced', default: None
#' Weights associated with classes in the form ``{class_label: weight}``.
#' If not given, all classes are supposed to have weight one.
#' The "balanced" mode uses the values of y to automatically adjust
#' weights inversely proportional to class frequencies in the input data
#' as ``n_samples / (n_classes * np.bincount(y))``.
#' Note that these weights will be multiplied with sample_weight (passed
#' through the fit method) if sample_weight is specified.
#' New in version 0.17: class_weight=’balanced’ instead of deprecated class_weight=’auto’.
#' @param max_iter int, default: 100
#' Useful only for the newton-cg, sag and lbfgs solvers.
#' Maximum number of iterations taken for the solvers to converge.
#' @param random_state int seed, RandomState instance, default: None
#' The seed of the pseudo random number generator to use when
#' shuffling the data. Used only in solvers 'sag' and 'liblinear'.
#' @param solver str {'newton-cg', 'lbfgs', 'liblinear', 'sag'}, default: 'liblinear'
#' Algorithm to use in the optimization problem.
#' - For small datasets, 'liblinear' is a good choice, whereas 'sag' is
#' faster for large ones.
#' - For multiclass problems, only 'newton-cg', 'sag' and 'lbfgs' handle
#' multinomial loss; 'liblinear' is limited to one-versus-rest
#' schemes.
#' - 'newton-cg', 'lbfgs' and 'sag' only handle L2 penalty.
#' Note that 'sag' fast convergence is only guaranteed on features with
#' approximately the same scale. You can preprocess the data with a
#' scaler from sklearn.preprocessing.
#' New in version 0.17: Stochastic Average Gradient descent solver.
#' @param tol float, default: 1e-4
#' Tolerance for stopping criteria.
#' @param multi_class str, {‘ovr’, ‘multinomial’}, default: ‘ovr’
#' Multiclass option can be either ‘ovr’ or ‘multinomial’. If the option chosen
#' is ‘ovr’, then a binary problem is fit for each label. Else the loss minimised
#' is the multinomial loss fit across the entire probability distribution. Works
#' only for the ‘newton-cg’, ‘sag’ and ‘lbfgs’ solver.
#' New in version 0.18: Stochastic Average Gradient descent solver for ‘multinomial’ case.
#' @param verbose int, default: 0
#' For the liblinear and lbfgs solvers set verbose to any positive number for verbosity.
#' @param warm_start bool, default: False
#' When set to True, reuse the solution of the previous call to fit as initialization,
#' otherwise, just erase the previous solution. Useless for liblinear solver.
#' New in version 0.17: warm_start to support lbfgs, newton-cg, sag solvers.
#' @param n_jobs int, default: 1
#' Number of CPU cores used during the cross-validation loop. If given a value
#' of -1, all cores are used.
#'
#' @name LogisticRegression
NULL

#' @rdname LogisticRegression
rsk_LogisticRegression <- R6::R6Class(
  "rsk_LogisticRegression",
  inherit = rsk_model,
  public = list(
    initialize = function(penalty, dual, C, fit_intercept, intercept_scaling,
                          class_weight, max_iter, random_state, solver, tol,
                          multi_class, verbose, warm_start, n_jobs){
      private$pointer <- sklearn$linear_model$LogisticRegression(
        penalty = penalty, dual = dual, C = C,
        fit_intercept = fit_intercept,
        intercept_scaling = intercept_scaling,
        class_weight = class_weight,
        max_iter = max_iter,
        random_state = random_state, solver = solver,
        tol = tol,
        multi_class = multi_class, verbose = verbose,
        warm_start = warm_start, n_jobs = n_jobs)
      private$pickle <- pickle$dumps(private$pointer)
    }
  )
)

#' @rdname LogisticRegression
#' @export
LogisticRegression <- function(x, y, penalty = "l2", dual = FALSE, C = 1.0,
                               fit_intercept = TRUE, intercept_scaling = 1.0,
                               class_weight = NULL, max_iter = 100,
                               random_state = NULL, solver = "liblinear",
                               tol = 1e-4, multi_class = "ovr",
                               verbose = 0, warm_start = FALSE, n_jobs = 1){

  model <- rsk_LogisticRegression$new(penalty = penalty, dual = dual, C = C,
                                      fit_intercept = fit_intercept,
                                      intercept_scaling = intercept_scaling,
                                      class_weight = class_weight,
                                      max_iter = max_iter,
                                      random_state = random_state, solver = solver,
                                      tol = tol,
                                      multi_class = multi_class, verbose = verbose,
                                      warm_start = warm_start, n_jobs = n_jobs)
  model$fit(x, y)
  return(model)
}

predict.rsk_LogisticRegression <- function(model, x, type = "class", ...){
  if(type == "class"){
    predictions <- model$predict(x)
  } else if (type == "log_proba"){
    predictions <- model$predict_log_proba(x)
  } else if (type == "proba"){
    predictions <- model$predict_proba(x)
  }
  return(predictions)
}

# Ridge -------------------------------------------------------------------

#' Ridge
#'
#' Linear least squares with l2 regularization.
#' This model solves a regression model where the loss function is the linear
#' least squares function and regularization is given by the l2-norm. Also known
#' as Ridge Regression or Tikhonov regularization. This estimator has built-in
#' support for multi-variate regression (i.e., when y is a 2d-array of shape
#' [n_samples, n_targets]).
#' Read more in the [User Guide](http://scikit-learn.org/stable/modules/linear_model.html#ridge-regression).
#'
#' @param alpha {float, array-like}, shape (n_targets)
#' Regularization strength; must be a positive float. Regularization
#' improves the conditioning of the problem and reduces the variance of
#' the estimates. Larger values specify stronger regularization.
#' Alpha corresponds to ``C^-1`` in other linear models such as
#' LogisticRegression or LinearSVC. If an array is passed, penalties are
#' assumed to be specific to the targets. Hence they must correspond in
#' number.
#' @param copy_X boolean, optional, default True
#' If True, X will be copied; else, it may be overwritten.
#' @param fit_intercept boolean
#' Whether to calculate the intercept for this model. If set
#' to false, no intercept will be used in calculations
#' (e.g. data is expected to be already centered).
#' max_iter : int, optional
#' Maximum number of iterations for conjugate gradient solver.
#' For 'sparse_cg' and 'lsqr' solvers, the default value is determined
#' by scipy.sparse.linalg. For 'sag' solver, the default value is 1000.
#' normalize : boolean, optional, default False
#' If True, the regressors X will be normalized before regression.
#' This parameter is ignored when `fit_intercept` is set to False.
#' When the regressors are normalized, note that this makes the
#' hyperparameters learnt more robust and almost independent of the number
#' of samples. The same property is not valid for standardized data.
#' However, if you wish to standardize, please use
#' `preprocessing.StandardScaler` before calling `fit` on an estimator
#' with `normalize=False`.
#' @param solver {'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag'}
#' Solver to use in the computational routines:
#'   - 'auto' chooses the solver automatically based on the type of data.
#' - 'svd' uses a Singular Value Decomposition of X to compute the Ridge
#' coefficients. More stable for singular matrices than
#' 'cholesky'.
#' - 'cholesky' uses the standard scipy.linalg.solve function to
#' obtain a closed-form solution.
#' - 'sparse_cg' uses the conjugate gradient solver as found in
#' scipy.sparse.linalg.cg. As an iterative algorithm, this solver is
#' more appropriate than 'cholesky' for large-scale data
#' (possibility to set `tol` and `max_iter`).
#' - 'lsqr' uses the dedicated regularized least-squares routine
#' scipy.sparse.linalg.lsqr. It is the fastest but may not be available
#' in old scipy versions. It also uses an iterative procedure.
#' - 'sag' uses a Stochastic Average Gradient descent. It also uses an
#' iterative procedure, and is often faster than other solvers when
#' both n_samples and n_features are large. Note that 'sag' fast
#' convergence is only guaranteed on features with approximately the
#' same scale. You can preprocess the data with a scaler from
#' sklearn.preprocessing.
#' All last four solvers support both dense and sparse data. However,
#' only 'sag' supports sparse input when `fit_intercept` is True.
#' @param tol float
#' Precision of the solution.
#' @param random_state int seed, RandomState instance, or None (default)
#' The seed of the pseudo random number generator to use when
#' shuffling the data. Used only in 'sag' solver.
#'
#' @name Ridge
NULL

#' @rdname Ridge
rsk_Ridge <- R6::R6Class(
  "rsk_Ridge",
  inherit = rsk_model,
  public = list(
    initialize = function(alpha, fit_intercept, normalize, copy_X, max_iter, tol,
                          solver, random_state){
      private$pointer <- sklearn$linear_model$Ridge(alpha = alpha,
                                                    fit_intercept = fit_intercept,
                                                    normalize = normalize,
                                                    copy_X = copy_X,
                                                    max_iter = max_iter,
                                                    tol = tol,
                                                    solver = solver,
                                                    random_state = random_state
                                                    )
      private$pickle <- pickle$dumps(private$pointer)
    }
  )
)

#' @rdname Ridge
#' @export
Ridge <- function(alpha=1.0, fit_intercept=TRUE, normalize=FALSE, copy_X=TRUE,
                  max_iter=NULL, tol=0.001, solver='auto', random_state=NULL){
  model <- rsk_Ridge$new(alpha = alpha,
                         fit_intercept = fit_intercept,
                         normalize = normalize,
                         copy_X = copy_X,
                         max_iter = max_iter,
                         tol = tol,
                         solver = solver,
                         random_state = random_state)
  model$fit(x, y)
  return(model)
}

predict.rsk_Ridge <- function(model, x, ...){
  model$predict(x)
}
