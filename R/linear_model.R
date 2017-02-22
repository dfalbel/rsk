rsk_LinearRegression <- R6::R6Class(
  "rsk_LinearRegression",
  inherit = rsk_model,
  public = list(
    initialize = function(...){
      self$pointer <- sklearn$linear_model$LinearRegression(...)
      self$pickle <- pickle$dumps(self$pointer)
    }
  )
)

LinearRegression <- function(...){
  model <- rsk_LinearRegression$new()
  model$fit(...)
  return(model)
}

predict.rsk_LinearRegression <- function(model, ...){
  model$predict(...)
}
