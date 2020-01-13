library(tidyverse)


x <- c(2,NA,NA,NA,NA,1,8,1,1,5,8,1,NA,3,NA,NA,NA,NA,5,2,1,NA,2,1,NA,2,1,1,1,1,1,2,1,NA,1,3)
mu <- mean(x, na.rm = T)
s <- sd(x, na.rm = T)



plot(table(x)/sum(ftable(x)), type = "h",
     xlim = c(0, 10),
     xlab = "iteartions",
     ylab = "",
     main = "Results found using our Search Engine")
lines(density(discard(x, is.na)))


perc.mean <- 1 - mean(is.na(x))
perc.sd <- perc.mean * (1 - perc.mean)

curve(dnorm(x, perc.mean, perc.sd),
      from = perc.mean - 3 * perc.sd,
      to   = perc.mean + 3 * perc.sd)

pnorm(0, perc.mean, perc.sd)

