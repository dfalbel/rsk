# LinearRegression --------------------------------------------------------

#' Linear Regression
#'
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
      self$pointer <- sklearn$linear_model$LinearRegression(fit_intercept, normalize, copy_X, n_jobs)
      self$pickle <- pickle$dumps(self$pointer)
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
#' @param x matrix. Training Data
#' @param y matrix. Target Values
#' @param penalty : str, 'l1' or 'l2', default: 'l2'
#' Used to specify the norm used in the penalization. The 'newton-cg',
#' 'sag' and 'lbfgs' solvers support only l2 penalties.
#' @param dual : bool, default: False
#' Dual or primal formulation. Dual formulation is only implemented for
#' l2 penalty with liblinear solver. Prefer dual=False when
#' n_samples > n_features.
#' @param C : float, default: 1.0
#' Inverse of regularization strength; must be a positive float.
#' Like in support vector machines, smaller values specify stronger
#' regularization.
#' @param fit_intercept : bool, default: True
#' Specifies if a constant (a.k.a. bias or intercept) should be
#' added to the decision function.
#' @param intercept_scaling : float, default 1.
#' Useful only when the solver 'liblinear' is used
#' and self.fit_intercept is set to True. In this case, x becomes
#' [x, self.intercept_scaling],
#' i.e. a "synthetic" feature with constant value equal to
#' intercept_scaling is appended to the instance vector.
#' The intercept becomes ``intercept_scaling * synthetic_feature_weight``.
#  Note! the synthetic feature weight is subject to l1/l2 regularization
#' as all other features.
#' To lessen the effect of regularization on synthetic feature weight
#' (and therefore on the intercept) intercept_scaling has to be increased.
#' @param class_weight : dict or 'balanced', default: None
#' Weights associated with classes in the form ``{class_label: weight}``.
#' If not given, all classes are supposed to have weight one.
#' The "balanced" mode uses the values of y to automatically adjust
#' weights inversely proportional to class frequencies in the input data
#' as ``n_samples / (n_classes * np.bincount(y))``.
#' Note that these weights will be multiplied with sample_weight (passed
#' through the fit method) if sample_weight is specified.
#' New in version 0.17: class_weight=’balanced’ instead of deprecated class_weight=’auto’.
#' @param max_iter : int, default: 100
#' Useful only for the newton-cg, sag and lbfgs solvers.
#' Maximum number of iterations taken for the solvers to converge.
#' @param random_state : int seed, RandomState instance, default: None
#' The seed of the pseudo random number generator to use when
#' shuffling the data. Used only in solvers 'sag' and 'liblinear'.
#' @param solver : {'newton-cg', 'lbfgs', 'liblinear', 'sag'}, default: 'liblinear'
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
#' @param tol : float, default: 1e-4
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
      self$pointer <- sklearn$linear_model$LogisticRegression(
        penalty = penalty, dual = dual, C = C,
        fit_intercept = fit_intercept,
        intercept_scaling = intercept_scaling,
        class_weight = class_weight,
        max_iter = max_iter,
        random_state = random_state, solver = solver,
        tol = tol,
        multi_class = multi_class, verbose = verbose,
        warm_start = warm_start, n_jobs = n_jobs)
      self$pickle <- pickle$dumps(self$pointer)
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

