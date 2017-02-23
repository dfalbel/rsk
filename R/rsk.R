sklearn <- NULL
pickle <- NULL

.onLoad <- function(libname, pkgname) {
  sklearn <<- reticulate::import("sklearn", delay_load = TRUE)
  pickle <<- reticulate::import("pickle", delay_load = TRUE)
}

rsk_model <- R6::R6Class(
  "rsk_model",
  public = list(
    fit = function(...){
      self$model$fit(...)
      private$pickle <- pickle$dumps(self$model)
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
      if(reticulate::py_is_null_xptr(private$pointer)){
        private$pointer <- pickle$loads(private$pickle)
      }
      return(private$pointer)
    }
  ),
  private = list(
    pointer = NULL,
    pickle = NULL
  )
)
