library(data.table)
library(cmdstanr)
library(rethinking)
library(yahoofinancer)
library(kernlab)
library(BART)
library(dbarts)
library(stats)
library(zoo)
library(ggplot2)
library(BayesTree)

PG <- Ticker$new("PG")

PG_data <- PG$get_history(
  start = "1980-03-03",
  interval = "1d"
)
df <- PG_data
df <- df[, c("date", "close", "volume", "high", "low", "open", "adj_close")]
cols <- c("close","volume","high","low","open","adj_close")
df <- na.omit(df)
df[cols] <- lapply(df[cols], function(x) {
  if (is.list(x)) x <- unlist(x)
  as.numeric(x)
})
df$date <- as.POSIXct(df$date)

df <- df[complete.cases(df), ]
df <- df[order(df$date), ]
df <- na.omit(df)
df$return <- c(NA, diff(log(df$close)))

df$lag_ret1 <- dplyr::lag(df$return, 1)
df$lag_ret2 <- dplyr::lag(df$return, 2)

df$ret3 <- log(df$close / dplyr::lag(df$close, 3))
df$ret5 <- log(df$close / dplyr::lag(df$close, 5))
df$ret10 <- log(df$close / dplyr::lag(df$close, 10))

df$vol5 <- rollapply(df$return, 5, sd, fill=NA, align="right")

df$ma5 <- stats::filter(df$close, rep(1/5,5), sides=1)
df$ma10 <- stats::filter(df$close, rep(1/10,10), sides=1)

df$volchg <- c(NA, diff(log(df$volume)))
df$vol10 <- rollapply(df$return, 10, sd, fill=NA, align="right")

df$momentum_vol <- df$ret5 * df$vol10

df$ma_ratio5  <- df$close / df$ma5
df$ma_ratio10 <- df$close / df$ma10

df$range <- df$high - df$low

df$lag1 <- c(NA, head(df$close, -1))
df$lag2 <- c(NA, NA, head(df$close, -2))
df$lag3 <- c(NA, NA, NA, head(df$close, -3))
df$lag4 <- c(NA, NA, NA, NA, head(df$close, -4))
df$lag5 <- c(NA, NA, NA, NA, NA, head(df$close, -5))
df$lag10 <- c(NA, NA, NA, NA, NA,NA, NA, NA, NA, NA, head(df$close, -10))
df$lag20 <- c(NA, NA, NA, NA, NA,NA, NA, NA, NA, NA,NA, NA, NA,
              NA, NA,NA, NA, NA, NA, NA, head(df$close, -20))

df$ret10 <- c(
  log(df$close[11:nrow(df)] / df$close[1:(nrow(df) - 10)]),
  rep(NA, 10)
)

df$y <- df$ret10
df$date <- as.Date(df$date)
df <- na.omit(df)

# Keep only rows with complete values in variables used by the model
df <- df[, c(
  "date","y","lag_ret1", "lag_ret2","ret3", 
  "ret5", "ret10","vol5", "vol10",
  "ma5", "ma10","ma_ratio5", "ma_ratio10","range",
  "volchg","momentum_vol", "lag1", "lag2", "lag3", "lag4", "lag5",
  "lag10", "lag20","volume", "open", "high", "low"
)]

df[] <- lapply(df, function(x) {
  if (is.list(x)) x <- unlist(x)
  x
})
df <- na.omit(df)
num_cols <- setdiff(names(df), "date")
df[num_cols] <- lapply(df[num_cols], as.numeric)


####SPlit Data
split_date <- as.Date("2026-03-03")

train <- df[df$date < split_date, ]
test  <- df[df$date >= split_date & df$date <= as.Date("2026-04-07"), ]

###Variables Included

X_train <- as.matrix(train[, c("lag_ret1", "lag_ret2","ret3", 
                               "ret5", "ret10","vol5", "vol10",
                               "ma5", "ma10","ma_ratio5", "ma_ratio10","range",
                               "volchg","momentum_vol", "lag1", "lag2", "lag3", "lag4", "lag5",
                               "lag10", "lag20","volume", "open", "high", "low")])
y_train <- as.numeric(train$y)

X_test  <- as.matrix(test[, c("lag_ret1", "lag_ret2","ret3", 
                              "ret5", "ret10","vol5", "vol10",
                              "ma5", "ma10","ma_ratio5", "ma_ratio10","range",
                              "volchg","momentum_vol", "lag1", "lag2", "lag3", "lag4", "lag5",
                              "lag10", "lag20","volume", "open", "high", "low")])
y_test  <- as.numeric(test$y)

storage.mode(X_train) <- "double"
storage.mode(X_test)  <- "double"

###BART Model
set.seed(999)

fit <- dbarts::bart(
  x.train = X_train,
  y.train = y_train,
  x.test = X_test,
  ntree = 200,
  ndpost = 400,
  nskip = 200,
  keeptrees = TRUE,
  verbose = FALSE
)

####Prediction vs test
yhat_train <- colMeans(fit$yhat.train)
yhat_test  <- colMeans(fit$yhat.test)

###RMSE
train_rmse <- sqrt(mean((y_train - yhat_train)^2))
test_rmse  <- sqrt(mean((y_test - yhat_test)^2))

train_rmse
test_rmse

rmse_SD <- sqrt(mean((y_test - 0)^2))
rmse_SD










##Predicting 5 days out
df$ret5 <- c(log(df$close[6:nrow(df)] / df$close[1:(nrow(df) - 5)]), rep(NA, 5))

df$y <- df$ret5
df$date <- as.Date(df$date)

# Keep only rows with complete values in variables used by the model
df <- df[, c("date", "y",
             "lag_ret1", "lag_ret2", "vol5", "ma10", "ma5", "range",
             "lag1", "lag2", "lag3", "lag4", "lag5",
             "volume", "open", "high", "low")]

df[] <- lapply(df, function(x) {
  if (is.list(x)) x <- unlist(x)
  x
})

num_cols <- setdiff(names(df), "date")
df[num_cols] <- lapply(df[num_cols], as.numeric)

df <- na.omit(df)

####SPlit Data
split_date <- as.Date("2026-03-03")

train <- df[df$date < split_date, ]
test  <- df[df$date >= split_date & df$date <= as.Date("2026-04-07"), ]

###Variables Included

X_train <- as.matrix(train[, c('lag_ret1','lag_ret2','vol5','ma10','ma5','range',
                               "lag1","lag2","lag3","lag4","lag5",
                               "volume","open","high","low")])
y_train <- as.numeric(train$y)

X_test  <- as.matrix(test[, c('lag_ret1','lag_ret2','vol5','ma10','ma5','range',
                              "lag1","lag2","lag3","lag4","lag5",
                              "volume","open","high","low")])
y_test  <- as.numeric(test$y)

storage.mode(X_train) <- "double"
storage.mode(X_test)  <- "double"

###BART Model
set.seed(999)

fit <- dbarts::bart(
  x.train = X_train,
  y.train = y_train,
  x.test = X_test,
  ntree = 100,
  ndpost = 500,
  nskip = 100,
  keeptrees = TRUE,
  verbose = FALSE
)

####Prediction vs test
yhat_train <- colMeans(fit$yhat.train)
yhat_test  <- colMeans(fit$yhat.test)

###RMSE
train_rmse <- sqrt(mean((y_train - yhat_train)^2))
test_rmse  <- sqrt(mean((y_test - yhat_test)^2))

train_rmse
test_rmse
rmse_SD <- sqrt(mean((y_test - 0)^2))
rmse_SD


####
#10 Days out
####
df$ret10 <- c(
  log(df$close[11:nrow(df)] / df$close[1:(nrow(df) - 10)]),
  rep(NA, 10)
)

df$y <- df$ret10
df$date <- as.Date(df$date)

# Keep only rows with complete values in variables used by the model
df <- df[, c("date", "y",
             "lag_ret1", "lag_ret2", "vol5", "ma10", "ma5", "range",
             "lag1", "lag2", "lag3", "lag4", "lag5",
             "volume", "open", "high", "low")]

df[] <- lapply(df, function(x) {
  if (is.list(x)) x <- unlist(x)
  x
})

num_cols <- setdiff(names(df), "date")
df[num_cols] <- lapply(df[num_cols], as.numeric)

df <- na.omit(df)

####SPlit Data
split_date <- as.Date("2026-03-03")

train <- df[df$date < split_date, ]
test  <- df[df$date >= split_date & df$date <= as.Date("2026-04-07"), ]

###Variables Included

X_train <- as.matrix(train[, c('lag_ret1','lag_ret2','vol5','ma10','ma5','range',
                               "lag1","lag2","lag3","lag4","lag5",
                               "volume","open","high","low")])
y_train <- as.numeric(train$y)

X_test  <- as.matrix(test[, c('lag_ret1','lag_ret2','vol5','ma10','ma5','range',
                              "lag1","lag2","lag3","lag4","lag5",
                              "volume","open","high","low")])
y_test  <- as.numeric(test$y)

storage.mode(X_train) <- "double"
storage.mode(X_test)  <- "double"

###BART Model
set.seed(999)

fit <- dbarts::bart(
  x.train = X_train,
  y.train = y_train,
  x.test = X_test,
  ntree = 5000,
  ndpost = 1000,
  nskip = 5000,
  keeptrees = TRUE,
  verbose = FALSE
)

####Prediction vs test
yhat_train <- colMeans(fit$yhat.train)
yhat_test  <- colMeans(fit$yhat.test)

###RMSE
train_rmse <- sqrt(mean((y_train - yhat_train)^2))
test_rmse  <- sqrt(mean((y_test - yhat_test)^2))

train_rmse
test_rmse
rmse_SD <- sqrt(mean((y_test - 0)^2))
rmse_SD


#########
#Results seemed too good wanted to try on a more volatile dataset
TSLA <- Ticker$new("TSLA")

TSLA_data <- TSLA$get_history(
  start = "2010-06-29",
  interval = "1d"
)
df <- TSLA_data
df <- df[, c("date", "close", "volume", "high", "low", "open", "adj_close")]
cols <- c("close","volume","high","low","open","adj_close")
df <- na.omit(df)
df[cols] <- lapply(df[cols], function(x) {
  if (is.list(x)) x <- unlist(x)
  as.numeric(x)
})
df$date <- as.POSIXct(df$date)

df <- df[complete.cases(df), ]
df <- df[order(df$date), ]
df <- na.omit(df)
df$return <- c(NA, diff(log(df$close)))

df$lag_ret1 <- dplyr::lag(df$return, 1)
df$lag_ret2 <- dplyr::lag(df$return, 2)

df$ret3 <- log(df$close / dplyr::lag(df$close, 3))
df$ret5 <- log(df$close / dplyr::lag(df$close, 5))
df$ret10 <- log(df$close / dplyr::lag(df$close, 10))

df$vol5 <- rollapply(df$return, 5, sd, fill=NA, align="right")

df$ma5 <- stats::filter(df$close, rep(1/5,5), sides=1)
df$ma10 <- stats::filter(df$close, rep(1/10,10), sides=1)

df$volchg <- c(NA, diff(log(df$volume)))
df$vol10 <- rollapply(df$return, 10, sd, fill=NA, align="right")

df$momentum_vol <- df$ret5 * df$vol10

df$ma_ratio5  <- df$close / df$ma5
df$ma_ratio10 <- df$close / df$ma10

df$range <- df$high - df$low

df$lag1 <- c(NA, head(df$close, -1))
df$lag2 <- c(NA, NA, head(df$close, -2))
df$lag3 <- c(NA, NA, NA, head(df$close, -3))
df$lag4 <- c(NA, NA, NA, NA, head(df$close, -4))
df$lag5 <- c(NA, NA, NA, NA, NA, head(df$close, -5))
df$lag10 <- c(NA, NA, NA, NA, NA,NA, NA, NA, NA, NA, head(df$close, -10))
df$lag20 <- c(NA, NA, NA, NA, NA,NA, NA, NA, NA, NA,NA, NA, NA,
              NA, NA,NA, NA, NA, NA, NA, head(df$close, -20))

df$ret10 <- c(
  log(df$close[11:nrow(df)] / df$close[1:(nrow(df) - 10)]),
  rep(NA, 10)
)

df$y <- df$ret10
df$date <- as.Date(df$date)
df <- na.omit(df)

# Keep only rows with complete values in variables used by the model
df <- df[, c(
  "date","y","lag_ret1", "lag_ret2","ret3", 
  "ret5", "ret10","vol5", "vol10",
  "ma5", "ma10","ma_ratio5", "ma_ratio10","range",
  "volchg","momentum_vol", "lag1", "lag2", "lag3", "lag4", "lag5",
  "lag10", "lag20","volume", "open", "high", "low"
)]

df[] <- lapply(df, function(x) {
  if (is.list(x)) x <- unlist(x)
  x
})
df <- na.omit(df)
num_cols <- setdiff(names(df), "date")
df[num_cols] <- lapply(df[num_cols], as.numeric)


####SPlit Data
split_date <- as.Date("2026-03-03")

train <- df[df$date < split_date, ]
test  <- df[df$date >= split_date & df$date <= as.Date("2026-04-07"), ]

###Variables Included

X_train <- as.matrix(train[, c("lag_ret1", "lag_ret2","ret3", 
                               "ret5", "ret10","vol5", "vol10",
                               "ma5", "ma10","ma_ratio5", "ma_ratio10","range",
                               "volchg","momentum_vol", "lag1", "lag2", "lag3", "lag4", "lag5",
                               "lag10", "lag20","volume", "open", "high", "low")])
y_train <- as.numeric(train$y)

X_test  <- as.matrix(test[, c("lag_ret1", "lag_ret2","ret3", 
                              "ret5", "ret10","vol5", "vol10",
                              "ma5", "ma10","ma_ratio5", "ma_ratio10","range",
                              "volchg","momentum_vol", "lag1", "lag2", "lag3", "lag4", "lag5",
                              "lag10", "lag20","volume", "open", "high", "low")])
y_test  <- as.numeric(test$y)

storage.mode(X_train) <- "double"
storage.mode(X_test)  <- "double"

###BART Model
set.seed(999)

fit <- dbarts::bart(
  x.train = X_train,
  y.train = y_train,
  x.test = X_test,
  ntree = 200,
  ndpost = 400,
  nskip = 200,
  keeptrees = TRUE,
  verbose = FALSE
)

####Prediction vs test
yhat_train <- colMeans(fit$yhat.train)
yhat_test  <- colMeans(fit$yhat.test)

###RMSE
train_rmse <- sqrt(mean((y_train - yhat_train)^2))
test_rmse  <- sqrt(mean((y_test - yhat_test)^2))

train_rmse
test_rmse

rmse_SD <- sqrt(mean((y_test - 0)^2))
rmse_SD








