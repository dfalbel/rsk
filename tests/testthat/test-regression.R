context("starting tests")

test_that("linear regression works", {

  model_rsk <- LinearRegression(as.matrix(mtcars[,c(3,4,5)]), mtcars$mpg)
  pred_rsk <- as.numeric(predict(model_rsk, as.matrix(mtcars[,c(3,4,5)])))

  model_lm <- lm(mpg ~ disp + hp + drat, data = mtcars)
  pred_lm <- as.numeric(predict(model_lm, mtcars))

  expect_equivalent(pred_rsk, pred_lm)
})
