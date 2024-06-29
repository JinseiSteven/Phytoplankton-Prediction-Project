
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
library(tidyverse)
library(readxl)
library(dplyr)
library(purrr)


# Set the maximum number of columns to display (e.g., 200)
options(width = 200)
options(max.print=1000000)



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

# drop 7 abio na rows, reorder IM
df_filtered <- df_filtered |> 
  filter(if_all("LOC_CODE": "IM [Jm2d]", ~!is.na(.))) |> 
  relocate("IM [Jm2d]", .before="DIN")
nrow(df_filtered)


abio <- df_filtered |> 
  select("LOC_CODE":"DIN:SI")
View(abio)

abio_cat <- abio |> 
  select(!LOC_CODE)


# # We can visually look for correlations between variables:
# abio.corr <- as.matrix(abs(cor(abio_cat)))
# View(abio.corr)
# 
# dev.new(width=7, height=7)
# heatmap(abs(abio.corr),
#         # Compute pearson correlation (note they are absolute values)
#         col = rev(heat.colors(100)),
#         Colv = NA, Rowv = NA, scale="none")
# legend("topright",
#        title = "Absolute Pearson R",
#        legend =  round(seq(0,1, length.out = 6),1),
#        y.intersp = 0.7, bty = "n",
#        fill = rev(heat.colors(6)))


rows_with_na <- sum(rowSums(is.na(abio)) > 0)
rows_with_na # should be 0

spec <- df_filtered |> 
  select("Agl":"Dno")

spec[is.na(spec)] <- 0
View(spec)


#factorize locations
clusters <- list(
  "Open Sea" = c('NOORDWK70', 'ROTTMPT50', 'ROTTMPT70', 'TERSLG100', 'TERSLG135', 'TERSLG175', 'TERSLG235'),
  "Offshore" = c('NOORDWK10', 'NOORDWK2', 'NOORDWK20', 'TERSLG10', 'TERSLG4', 'WALCRN20', 'WALCRN70'),
  "Coastal" = c('DANTZGT', 'GOERE6', 'HUIBGOT', 'LODSGT', 'MARSDND', 'ROTTMPT3', 'VLISSGBISSVH', 'WALCRN2'),
  "Coastal/Estuary" = c('HANSWGL'),
  "Real Estuaries" = c('GROOTGND', 'SCHAARVODDL'),
  "Lakes" = c('DREISR', 'SOELKKPDOT')
)

abio <- abio |> 
  mutate(cluster = case_when(
    LOC_CODE %in% clusters[["Open Sea"]] ~ "Open Sea",
    LOC_CODE %in% clusters[["Offshore"]] ~ "Offshore",
    LOC_CODE %in% clusters[["Coastal"]] ~ "Coastal",
    LOC_CODE %in% clusters[["Coastal/Estuary"]] ~ "Coastal/Estuary",
    LOC_CODE %in% clusters[["Real Estuaries"]] ~ "Real Estuaries",
    LOC_CODE %in% clusters[["Lakes"]] ~ "Lakes",
    TRUE ~ "Other"  # This line is optional, it assigns "Other" to any locations not found in the clusters
  )) |> 
  mutate(cluster  = as.factor(cluster))


#Hellinger transform for spec
spec.hel <- decostand(spec, "hellinger")
View(spec.hel)

abio.z <- abio |> 
  select(`ZS [mg/l]`:`DIN:SI`)
View(abio.z)

# standardize
# Scale and center variables
abio.z <- decostand(abio.z, method = "standardize")
# Variables are now centered around a mean of 0
round(apply(abio.z, 2, mean), 1)

# and scaled to have a standard deviation of 1
apply(abio.z, 2, sd)


abio_std <- cbind(cluster = abio$cluster, abio.z)
# abio_std$loc <- as.factor(abio_std$loc)
View(abio_std)

(spe.rda <- rda(spec.hel ~ ., abio_std))
summary(spe.rda)

# Unadjusted R^2 retrieved from the rda object
(R2 <- RsquareAdj(spe.rda)$r.squared)
# Adjusted R^2 retrieved from the rda object
(R2adj <- RsquareAdj(spe.rda)$adj.r.squared)


dev.new(height=7,width=7)
plot(spe.rda,
     scaling=1,
     # type="text",
     display=c("sp", "lc", "cn"),
     main="Triplot RDA spe.hel ~env3 ~scaling 1 - lc scores"
     )
spe.sc1 <- 
  scores(spe.rda,
         choices=1:2,
         scaling=1,
         display="sp")

arrows(0, 0,
       spe.sc1[,1] * .92,
       spe.sc1[,2] * 0.92,
       length=0,
       lty=1,
       col="red")

View(abio_std)

# ## Global test of the RDA result
# anova(spe1.rda, permutations = how(nperm = 999))
# ## Tests of all canonical axes
# anova(spe1.rda, by = "axis", permutations = how(nperm = 999))
# 
# anova(spe1.rda, by = "term", permutations = how(nperm = 999))



spechem.physio <- 
  rda(spec.hel ~ cluster, data=abio_std)

summary(spechem.physio)


dev.new(width=10, height=10)
plot(spechem.physio,
     scaling=2,
     type="text",
     display=c("sp", "lc", "cn"),
     main="Triplot RDA spe.hel ~env3 ~scaling 1 - lc scores"
)
spe.sc1 <- 
  scores(spechem.physio,
         choices=1:2,
         scaling=2,
         display="sp")

arrows(0, 0,
       spe.sc1[,1] * .92,
       spe.sc1[,2] * 0.92,
       length=0,
       lty=1,
       col="red")

#ordistep
# RDA with all explanatory variables except cluster
abio_std
no_cluster <- abio_std[2:ncol(abio_std)]
speclust.rda <- rda(spec.hel ~ ., data = no_cluster)


dev.new(width=10, height=10)
plot(speclust.rda,
     scaling=2,
     type="text",
     display=c("sp", "lc", "cn"),
     main="Triplot RDA spe.hel ~env3 ~scaling 1 - lc scores"
)
spe.sc1 <- 
  scores(speclust.rda,
         choices=1:2,
         scaling=2,
         display="sp")

arrows(0, 0,
       spe.sc1[,1] * .92,
       spe.sc1[,2] * 0.92,
       length=0,
       lty=1,
       col="red")


library(plotly)

source("rda_plots.R")

scaling <- 2

# Perform RDA and extract necessary scores
spe.rda <- rda(spec.hel ~ `SALNTT [DIMSLS]` + `T [oC]` + kPAR_14d + `CHLFa [ug/l]` +      `ZS [mg/l]` + `SiO2 [umol/L]` + `NO3 [umol/L]` + `NO2 [umol/L]` +      `pH [DIMSLS]` + DIN + `E [/m]` + `DIN:SRP` + `PO4 [umol/L]` +      `NH4 [umol/L]`, abio_std)
perc <- round(100 * (summary(spe.rda)$cont$importance[2, 1:3]), 2)
sc_sp <- scores(spe.rda, display = "species", choices = c(1, 2, 3), scaling = scaling)
sc_bp <- scores(spe.rda, display = "bp", choices = c(1, 2, 3), scaling = scaling)







spec_clusters <- list(
  "Harmful Species" = c('Fja', 'Cha', 'Dac', 'Pmi')
)



sc_sp <- as.data.frame(sc_sp)
sc_sp <- sc_sp |> 
  mutate(spec_cluster = case_when(
    row.names(sc_sp) %in% unlist(spec_clusters[["Harmful Species"]]) ~ "Harmful Species",
    TRUE ~ "Other"  # This line is optional, it assigns "Other" to any species not found in the clusters
  )) |> 
  mutate(spec_cluster = as.factor(spec_cluster))

print(sc_sp)
sc_bp <- scores(spe.rda, display = "bp", choices = c(1, 2, 3), scaling = scaling)
plot_3d_rda(sc_sp, sc_bp, palette, perc, clusterdef=sc_sp$spec_cluster)


plot_3d_rda(sc_sp, sc_bp, palette, perc)



# Create the plot
p <- plot_ly() |>
  # Add species points
  add_trace(
    type = "scatter",
    mode = "markers",
    x = sc_sp[, 1],
    y = sc_sp[, 2],
    text = rownames(sc_sp),
    textposition = "top center",
    marker = list(symbol = "circle", size = 10, color = "#f2bd33"),
    name = "Species"
  )

# Add arrows for explanatory variables
for (i in 1:nrow(sc_bp)) {
  p <- p |>
    add_trace(
      type = "scatter",
      mode = "lines+text",
      x = c(0, sc_bp[i, 1]),
      y = c(0, sc_bp[i, 2]),
      line = list(color = "red", width = 3),
      text = rownames(sc_bp)[i],
      textposition = "top center",
      name = rownames(sc_bp)[i],
      showlegend = TRUE
    )
}

# Add arrowheads and large text using annotations
annotations <- list()
for (i in 1:nrow(sc_bp)) {
  annotations[[i]] <- list(
    x = sc_bp[i, 1],
    y = sc_bp[i, 2],
    ax = 0,
    ay = 0,
    xref = "x",
    yref = "y",
    axref = "x",
    ayref = "y",
    showarrow = TRUE,
    arrowhead = 2,
    arrowsize = 1,
    arrowwidth = 2,
    arrowcolor = "red"
  )
}

# Customize layout and add annotations
p <- p |>
  layout(
    title = "Triplot RDA - scaling 2",
    xaxis = list(title = paste0("RDA1 (", perc[1], "%)")),
    yaxis = list(title = paste0("RDA2 (", perc[2], "%)")),
    showlegend = FALSE,
    annotations = annotations
  )

p



# # forward selection of variables
# 
# mod0 <- rda(spec.hel ~ 1, data=no_cluster)
# step.forward <-
#   ordiR2step(mod0,
#              scope=formula(spe.rda.all),
#              direction="forward",
#              prumtations=how(nperm=499))
# RsquareAdj(step.forward)