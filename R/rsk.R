sklearn <- NULL
pickle <- NULL

.onLoad <- function(libname, pkgname) {
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
