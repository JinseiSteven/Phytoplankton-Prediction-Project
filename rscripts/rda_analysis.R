
# for rda
library(ade4)
library(adegraphics)
library(adespatial)
library(vegan)
library(vegan3d)
library(MASS)
library(ellipse)
library(FactoMineR)
library(rrcov)
source("hcoplot.R")
source("triplot.rda.R")
source("plot.lda.R")
source("polyvars.R")
source("screestick.R")

# general utility
library(readxl)
library(dplyr)


# Set the maximum number of columns to display (e.g., 200)
options(width = 200)
df <- read_excel("MERGED_DATA_INTERPOLATED_FINAL.xlsx", sheet="MERGE")
View(df)

n_species <-  df |>
  select("Agl":"Dno") |> 
  ncol()
n_species

# checks to see how many only nan values for species we have
allnas_cond <- rowSums(is.na(select(df, "Agl":"Dno"))) == n_species
allnas_rows <- df |> 
  filter(allnas_cond)
nrow(allnas_rows)

# retains only rows where species were actually measured
df_filtered <- df |> filter(!allnas_cond)
nrow(df_filtered)

# drops TIJD (not accurate, incomplete), ZICHT (incomplete),
# DATUM (irrelevant unless transformed?),  
# E_method (irrelevant), IM (need to FIX!!!!!)
df_filtered <- df_filtered |> 
  select(!c(
    "TIJD", "ZICHT [dm]",
    "DATUM",
    "E_method"))

# drop 7 abio na rows
df_filtered <- df_filtered |> 
  filter(if_all("LOC_CODE": "IM [Jm2d]", ~!is.na(.)))
nrow(df_filtered)

# should be "IM [Jm2d]" !!!!!
abio <- df_filtered |> 
  select("LOC_CODE":"IM [Jm2d]")
View(abio)

abio_cat <- abio |> 
  select(!LOC_CODE)

# # We can visually look for correlations between variables:
# dev.new(width=7, height=7)
# heatmap(abs(cor(abio_cat)), 
#         # Compute pearson correlation (note they are absolute values)
#         col = rev(heat.colors(6)), 
#         Colv = NA, Rowv = NA)
# legend("topright", 
#        title = "Absolute Pearson R",
#        legend =  round(seq(0,1, length.out = 6),1),
#        y.intersp = 0.7, bty = "n",
#        fill = rev(heat.colors(6)))
# 


rows_with_na <- sum(rowSums(is.na(abio)) > 0)
rows_with_na # should be 0

spec <- df_filtered |> 
  select("Agl":"Dno")

spec[is.na(spec)] <- 0
View(spec)

#Hellinger transform for spec
spec.hel <- decostand(spec, "hellinger")
View(spec.hel)

(spec.rda <- rda(spec.hel ~ ., abio))
summary(spec.rda)

# Unadjusted R^2 retrieved from the rda object
(R2 <- RsquareAdj(spec.rda)$r.squared)
# Adjusted R^2 retrieved from the rda object
(R2adj <- RsquareAdj(spec.rda)$adj.r.squared)



# Alternatively, you can set a higher or lower threshold depending on the number of species you want to display

# Plot the RDA results with scaling 1
dev.new(width=7, height=7)
plot(spec.rda, 
     scaling = 2, 
     display = c("sp", "lc", "cn"), 
     main = "Triplot RDA spec.hel ~ abio - scaling 1 - lc scores")

# Add arrows for the species, only displaying those selected
spe.sc1 <- scores(spec.rda, choices = 1:2, scaling = 1, display = "sp")
arrows(0, 0, spe.sc1[sel.sp, 1] * 0.92, spe.sc1[sel.sp, 2] * 0.92, length = 0, lty = 1, col = "red")

# Add labels for the selected species
text(spe.sc1[sel.sp, 1] * 0.92, spe.sc1[sel.sp, 2] * 0.92, labels = rownames(spe.sc1)[sel.sp], col = "red", cex = 0.7)
# 
# 
# # Plot the RDA results with scaling 2
# dev.new(width=7, height=7)
# plot(spec.rda, 
#      scaling = 2, 
#      display = c("sp", "lc", "cn"), 
#      main = "Triplot RDA spec.hel ~ abio - scaling 2 - lc scores")
# 
# # Add arrows for the species, only displaying those selected
# spec.sc2 <- scores(spec.rda, choices = 1:2, scaling = 2, display = "sp")
# arrows(0, 0, spec.sc2[sel.sp, 1] * 0.92, spec.sc2[sel.sp, 2] * 0.92, length = 0, lty = 1, col = "red")
# 
# # Add labels for the selected species
# text(spec.sc2[sel.sp, 1] * 0.92, spec.sc2[sel.sp, 2] * 0.92, labels = rownames(spec.sc2)[sel.sp], col = "red", cex = 0.7)


