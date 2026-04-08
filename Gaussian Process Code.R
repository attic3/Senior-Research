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

VOO <- Ticker$new("VOO")

VOO_data <- VOO$get_history(start = "2006-3-3", interval = "1d")

# Response: use adjusted close prices
y <- as.matrix(VOO_data$adj_close)

# Input: use time index
x <- matrix(1:nrow(VOO_data), ncol = 1)

# Number of observations
m <- nrow(x)

m_x <- matrix(0, nrow = m, ncol = 1)


sigma_f <- 1
ell <- 20

k <- function(xi, xj, sigma_f, ell) {
  sigma_f^2 * exp(- (xi - xj)^2 / (2 * ell^2))
}



K <- matrix(0, nrow = m, ncol = m)

K

for (i in 1:m) {
  for (j in 1:m) {
    K[i, j] <- k(x[i], x[j], sigma_f, ell)
  }
}



sigma_n <- 0.1
Ky <- K + sigma_n^2 * diag(m)

Ky


y[1:5, , drop = FALSE]      # first few f(x_i)
m_x[1:5, , drop = FALSE]    # first few m(x_i)
K[1:5, 1:5]                 # first part of covariance matrix


install.packages("mvtnorm")
library(mvtnorm)

y <- as.numeric(VOO_data$adj_close)

dmvnorm(y,
        mean = as.vector(m_x),
        sigma = Ky)


K_star <- K
Ky_inv <- solve(Ky)

mu_post <- m_x + K_star %*% Ky_inv %*% (y - m_x)



step1 <- y - m_x
step2 <- solve(Ky)
step3 <- step2 %*% step1
step4 <- K %*% step3
mu_post <- m_x + step4



Sigma_post <- K - K %*% solve(Ky) %*% K

sd_post <- sqrt(diag(Sigma_post))


plot(x, y, pch = 1, main = "GP Fit on VOO Adjusted Close",
     xlab = "Time Index", ylab = "Adjusted Close")
lines(x, mu_post, lwd = 0.2)


mu_post <- as.vector(mu_post)

plot(x, y, pch = 16,
     main = "GP Fit on VOO Adjusted Close",
     xlab = "Time Index", ylab = "Adjusted Close")

lines(as.vector(x), as.vector(mu_post), lwd = 2)

head(mu_post)
length(mu_post)

plot(as.vector(x), y, pch = 16, cex = 0.4,
     main = "GP Fit on VOO Adjusted Close",
     xlab = "Time Index", ylab = "Adjusted Close")

lines(as.vector(x), as.vector(mu_post), lwd = 2)


plot(as.vector(x), y, pch = 16, cex = 0.1,
     main = "GP Fit on VOO Adjusted Close",
     xlab = "Time Index", ylab = "Adjusted Close")
lines(as.vector(x), as.vector(mu_post), lwd = 2)


mean_col <- "blue"

upper_col <- adjustcolor("red", alpha.f = 0.3)
lower_col <- adjustcolor("red", alpha.f = 0.3)


plot(as.vector(x), y, pch = 16, cex = 0.09,
     main = "GP Fit on VOO Adjusted Close",
     xlab = "Time Index", ylab = "Adjusted Close")




polygon(
  c(as.vector(x), rev(as.vector(x))),
  c(mu_post + 2 * mu_post, rev(mu_post - 2 * sd_post)),
  col = adjustcolor("blue", alpha.f = 0.2),
  border = NA
)

lines(as.vector(x), as.vector(mu_post), col = "blue", lwd = 2)

dmvnorm(y, mean = m_x, sigma = Ky, log = TRUE)



#train and test model

VOO_data$date <- as.Date(VOO_data$date)

train_data <- subset(VOO_data, date <= as.Date("2026-03-03"))


y <- as.numeric(train_data$adj_close)
x <- matrix(1:length(y), ncol = 1)

m <- length(y)
m_x <- rep(0, m)
y_mat <- matrix(y, ncol = 1)
m_x_mat <- matrix(m_x, ncol = 1)


sigma_f <- 1
ell <- 20
sigma_n <- 0.1

k <- function(xi, xj, sigma_f, ell) {
  sigma_f^2 * exp(-(xi - xj)^2 / (2 * ell^2))
}

K <- matrix(0, nrow = m, ncol = m)

for (i in 1:m) {
  for (j in 1:m) {
    K[i, j] <- k(x[i], x[j], sigma_f, ell)
  }
}

Ky <- K + sigma_n^2 * diag(m)

x_star <- matrix(m + 1, ncol = 1)

K_star <- matrix(0, nrow = 1, ncol = m)

for (j in 1:m) {
  K_star[1, j] <- k(x_star[1], x[j], sigma_f, ell)
}

K_starstar <- matrix(k(x_star[1], x_star[1], sigma_f, ell), nrow = 1, ncol = 1)


Ky_inv <- solve(Ky)

mu_star <- K_star %*% Ky_inv %*% (y_mat - m_x_mat)
Sigma_star <- K_starstar - K_star %*% Ky_inv %*% t(K_star)

mu_star
Sigma_star

sd_star <- sqrt(Sigma_star[1,1])
mu_next <- mu_star[1,1]


close_grid <- seq(mu_next - 4 * sd_star, mu_next + 4 * sd_star, length.out = 500)
densities <- dnorm(close_grid, mean = mu_next, sd = sd_star)

plot(close_grid, densities, type = "l", lwd = 2,
     main = "Predictive Density for Next-Day VOO Close",
     xlab = "Possible Next-Day Close",
     ylab = "Density")
abline(v = mu_next, lty = 2)


pnorm(525, mean = mu_next, sd = sd_star) -
  pnorm(520, mean = mu_next, sd = sd_star)


1 - pnorm(525, mean = mu_next, sd = sd_star)


breaks <- seq(mu_next - 4 * sd_star, mu_next + 4 * sd_star, by = 1)

bin_probs <- numeric(length(breaks) - 1)

for (i in 1:(length(breaks) - 1)) {
  bin_probs[i] <- pnorm(breaks[i + 1], mean = mu_next, sd = sd_star) -
    pnorm(breaks[i], mean = mu_next, sd = sd_star)
}

data.frame(
  lower = breaks[-length(breaks)],
  upper = breaks[-1],
  probability = bin_probs
)






####FUll
# If needed
# install.packages("mvtnorm")

library(mvtnorm)

# -----------------------------
# 1. Prepare the data
# -----------------------------

# Make sure date is a Date object
VOO_data$date <- as.Date(VOO_data$date)

# Keep data through March 3, 2026
train_data <- subset(VOO_data, date <= as.Date("2026-03-03"))

# Use CLOSE price, not adjusted close
y <- as.numeric(train_data$close)

# Input variable: time index
x <- matrix(1:length(y), ncol = 1)

# Number of observations
m <- length(y)

# Mean function: zero mean
m_x <- rep(0, m)

# Matrix versions for explicit matrix multiplication
y_mat <- matrix(y, ncol = 1)
m_x_mat <- matrix(m_x, ncol = 1)

# -----------------------------
# 2. Define the kernel
# -----------------------------

sigma_f <- 1
ell <- 20
sigma_n <- 0.1

k <- function(xi, xj, sigma_f, ell) {
  sigma_f^2 * exp(-(xi - xj)^2 / (2 * ell^2))
}

# -----------------------------
# 3. Build the covariance matrix K
# -----------------------------

K <- matrix(0, nrow = m, ncol = m)

for (i in 1:m) {
  for (j in 1:m) {
    K[i, j] <- k(x[i], x[j], sigma_f, ell)
  }
}

# Add noise variance to the diagonal
Ky <- K + sigma_n^2 * diag(m)

# Optional small jitter for numerical stability
Ky <- Ky + 1e-6 * diag(m)

# -----------------------------
# 4. Define the next-day input x*
# -----------------------------

# The next trading day after the training sample
x_star <- matrix(m + 1, nrow = 1, ncol = 1)

# -----------------------------
# 5. Build K_star and K_starstar
# -----------------------------

# K_star = covariance between next day and training points
K_star <- matrix(0, nrow = 1, ncol = m)

for (j in 1:m) {
  K_star[1, j] <- k(x_star[1], x[j], sigma_f, ell)
}

# K_starstar = covariance of next day with itself
K_starstar <- matrix(k(x_star[1], x_star[1], sigma_f, ell), nrow = 1, ncol = 1)

# -----------------------------
# 6. GP predictive mean and variance
# -----------------------------

Ky_inv <- solve(Ky)

# Posterior mean for next day
mu_star <- K_star %*% Ky_inv %*% (y_mat - m_x_mat)

# Posterior variance for next day
Sigma_star <- K_starstar - K_star %*% Ky_inv %*% t(K_star)

# Extract scalar values
mu_next <- mu_star[1, 1]
var_next <- Sigma_star[1, 1]
sd_next <- sqrt(var_next)

# Print prediction summary
mu_next
var_next
sd_next

# -----------------------------
# 7. Predictive density plot
# -----------------------------

close_grid <- seq(mu_next - 4 * sd_next, mu_next + 4 * sd_next, length.out = 500)
densities <- dnorm(close_grid, mean = mu_next, sd = sd_next)

plot(close_grid, densities, type = "l", lwd = 2,
     main = "Predictive Density for Next-Day VOO Close",
     xlab = "Possible next-day close",
     ylab = "Density")
abline(v = mu_next, lty = 2)

# -----------------------------
# 8. Probability of intervals
# -----------------------------

# Example: probability next close is between 520 and 525
prob_520_525 <- pnorm(525, mean = mu_next, sd = sd_next) -
  pnorm(520, mean = mu_next, sd = sd_next)

# Example: probability next close is above 525
prob_above_525 <- 1 - pnorm(525, mean = mu_next, sd = sd_next)

prob_520_525
prob_above_525

# -----------------------------
# 9. Probabilities across many close-price bins
# -----------------------------

# Create $1 bins around the predictive mean
breaks <- seq(floor(mu_next - 4 * sd_next),
              ceiling(mu_next + 4 * sd_next),
              by = 1)

bin_probs <- numeric(length(breaks) - 1)

for (i in 1:(length(breaks) - 1)) {
  bin_probs[i] <- pnorm(breaks[i + 1], mean = mu_next, sd = sd_next) -
    pnorm(breaks[i], mean = mu_next, sd = sd_next)
}

prob_table <- data.frame(
  lower_close = breaks[-length(breaks)],
  upper_close = breaks[-1],
  probability = bin_probs
)

prob_table

# -----------------------------
# 10. Most likely close interval
# -----------------------------

prob_table[which.max(prob_table$probability), ]




#############################################################

breaks <- seq(floor(mu_next - 2 * sd_next),
              ceiling(mu_next + 2 * sd_next),
              by = 0.1)

bin_probs <- numeric(length(breaks) - 1)

for (i in 1:(length(breaks) - 1)) {
  bin_probs[i] <- pnorm(breaks[i + 1], mean = mu_next, sd = sd_next) -
    pnorm(breaks[i], mean = mu_next, sd = sd_next)
}

prob_table <- data.frame(
  lower_close = breaks[-length(breaks)],
  upper_close = breaks[-1],
  probability = bin_probs
)

prob_table

