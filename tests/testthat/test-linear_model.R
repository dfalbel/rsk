context("linear_model")

test_that("LinearRegression", {

  model_rsk <- LinearRegression(as.matrix(mtcars[,c(3,4,5)]), mtcars$mpg)
  pred_rsk <- as.numeric(predict(model_rsk, as.matrix(mtcars[,c(3,4,5)])))

  model_lm <- lm(mpg ~ disp + hp + drat, data = mtcars)
  pred_lm <- as.numeric(predict(model_lm, mtcars))

  expect_equivalent(pred_rsk, pred_lm)
})

test_that("LogisticRegression", {

  model_rsk <- LogisticRegression(as.matrix(iris[,1:2]), iris$Species == "setosa")
  pred_rsk <- predict(model_rsk, as.matrix(iris[,1:2]))

  expect_length(pred_rsk, 150)
})

test_that("Ridge", {

  model_rsk <- Ridge(as.matrix(mtcars[,c(3,4,5)]), mtcars$mpg)
  pred_rsk <- as.numeric(predict(model_rsk, as.matrix(mtcars[,c(3,4,5)])))

  expect_length(pred_rsk, 32)
})
