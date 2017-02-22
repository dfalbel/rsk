# python 'foo' module I want to use in my package
sklearn <- NULL
pickle <- NULL

.onLoad <- function(libname, pkgname) {
  # delay load foo module (will only be loaded when accessed via $)
  sklearn <<- reticulate::import("sklearn", delay_load = TRUE)
  pickle <<- reticulate::import("pickle", delay_load = TRUE)
}

rsk_model <- R6::R6Class(
  "rsk_model",
  public = list(
    pointer = NULL,
    pickle = NULL,
    fit = function(...){
      self$model$fit(...)
      self$pickle <- pickle$dumps(self$pointer)
    },
    predict = function(...){
      self$model$predict(...)
    },
    print = function(...){
      print(self$model)
    }
  ),
  active = list(
    model = function(){
      if(reticulate::py_is_null_xptr(self$pointer)){
        self$pointer <- pickle$loads(self$pickle)
      }
      return(self$pointer)
    }
  )
  )

rsk_LinearRegression <- R6::R6Class(
  "sk.linear_model.LinearRegression",
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

predict.rsk_model <- function(model, ...){
  model$predict(...)
}
