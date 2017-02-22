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
#' @export
LinearRegression <- function(x, y, fit_intercept = TRUE, normalize = FALSE,
                             copy_X = TRUE, n_jobs = 1){
  model <- rsk_LinearRegression$new(fit_intercept, normalize, copy_X, n_jobs)
  model$fit(x, y)
  return(model)
}

predict.rsk_LinearRegression <- function(model, ...){
  model$predict(...)
}
