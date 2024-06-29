
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
library(plotly)
library(RColorBrewer)
source("hcoplot.R")
source("triplot.rda.R")
source("plot.lda.R")
source("polyvars.R")
source("screestick.R")
source("rda_plots.R")

# general utility
library(tidyverse)
library(readxl)
library(dplyr)
library(purrr)



# Set the maximum number of columns to display (e.g., 200)
options(width = 200)
options(max.print=1000)



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

#factorize locations
clusters <- list(
  "Open Sea" = c('NOORDWK70', 'ROTTMPT50', 'ROTTMPT70', 'TERSLG100', 'TERSLG135', 'TERSLG175', 'TERSLG235'),
  "Offshore" = c('NOORDWK10', 'NOORDWK2', 'NOORDWK20', 'TERSLG10', 'TERSLG4', 'WALCRN20', 'WALCRN70'),
  "Coastal" = c('DANTZGT', 'GOERE6', 'HUIBGOT', 'LODSGT', 'MARSDND', 'ROTTMPT3', 'VLISSGBISSVH', 'WALCRN2'),
  "Coastal-Estuary" = c('HANSWGL'),
  "Real Estuaries" = c('GROOTGND', 'SCHAARVODDL'),
  "Lakes" = c('DREISR', 'SOELKKPDOT')
)

df_filtered <- df_filtered |> 
  mutate(cluster = case_when(
    LOC_CODE %in% clusters[["Open Sea"]] ~ "Open Sea",
    LOC_CODE %in% clusters[["Offshore"]] ~ "Offshore",
    LOC_CODE %in% clusters[["Coastal"]] ~ "Coastal",
    LOC_CODE %in% clusters[["Coastal-Estuary"]] ~ "Coastal-Estuary",
    LOC_CODE %in% clusters[["Real Estuaries"]] ~ "Real Estuaries",
    LOC_CODE %in% clusters[["Lakes"]] ~ "Lakes",
    TRUE ~ "Other"  # This line is optional, it assigns "Other" to any locations not found in the clusters
  )) |> 
  mutate(cluster= as.factor(cluster)) |> 
  relocate(cluster, .before=`ZS [mg/l]`)
View(df_filtered)

groups <- split(df_filtered, df_filtered$cluster)

abio <- lapply(groups, function(df){
  df |> select("LOC_CODE":"DIN:SI")
})
  
View(abio)

# abio_cat <- abio |> 
#   select(!LOC_CODE)


# We can visually look for correlations between variables:
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

# makes spec groups
spec <- lapply(groups, function(df){
  df |> 
    select("Agl":"Dno") |>
    mutate_all(~replace(., is.na(.), 0))
})
View(spec[["Coastal"]])

#Hellinger transform for spec
spec.hel <- lapply(spec, function(df) {
  decostand(df, "hellinger")
}) 
  

standardize_vars <- function(df) {
  df <- df |> 
    select(`ZS [mg/l]`:`DIN:SI`) |> 
    decostand(method="standardize")
}

abio.z <- lapply(abio, standardize_vars)


# # Variables are now centered around a mean of 0
# round(apply(abio.z, 2, mean), 1)
# 
# # and scaled to have a standard deviation of 1
# apply(abio.z, 2, sd)


# Assuming 'groups' is your list of data frames and 'abio.z' is your list of standardized tibbles
abio_std <- lapply(seq_along(groups), function(i) {
  cbind(loc = groups[[i]]$LOC_CODE, abio.z[[i]]) |> 
    transform(loc = as.factor(loc))
})
names(abio_std) <- names(groups)

View(abio_std[["Coastal"]])




palette <- brewer.pal(10, "Paired")
perform_rda <- function(group_name, scaling=2) {
  spe.rda <- rda(spec.hel[[group_name]] ~ ., abio_std[[group_name]])
  print(summary(spe.rda))
  perc <- round(100 * (summary(spe.rda)$cont$importance[2, 1:3]), 2)
  sc_sp <- scores(spe.rda, display = "species", choices = c(1, 2, 3), scaling = scaling)
  print(sc_sp)
  sc_bp <- scores(spe.rda, display = "bp", choices = c(1, 2, 3), scaling = scaling)
  plot_3d_rda(sc_sp, sc_bp, palette, perc)
  # Unadjusted R^2 retrieved from the rda object
  # (R2 <- RsquareAdj(spe.rda)$r.squared)
  # Adjusted R^2 retrieved from the rda object
  # (R2adj <- RsquareAdj(spe.rda)$adj.r.squared)
  # spe.good <- goodness(spe.rda)
  # sel.sp <- which(spe.good[, 2] >= 0.05)
  # print(paste("Significance for", group_name))
  # print(anova(spe.rda, by = "term", permutations = how(nperm = 999)))
}

perform_rda("Open Sea", scaling=2)

# env_names <- names(spec.hel)
# for (env in env_names) {
#   print(paste("Performing RDA for", env))
#   perform_rda(env)
# }






scaling <- 2

spe.rda <- rda(spec.hel[["Coastal"]] ~ . + Condition(loc), abio_std[["Coastal"]])
print(summary(spe.rda))
perc <- round(100 * (summary(spe.rda)$cont$importance[2, 1:3]), 2)
sc_sp <- scores(spe.rda, display = "species", choices = c(1, 2, 3), scaling = scaling)

spec_clusters <- list(
  "cluster_0" = c('Agl', 'Ezo', 'Gde', 'Osi', 'Rse', 'Gfl', 'Gsp', 'Nsc', 'Pbi', 'Pha', 'Kgl', 'Lan', 'Pmi', 'Cdeb', 'Ptr'),
  "cluster_1" = c('Dbr', 'Omo', 'Orh', 'Tec', 'Tro', 'Dle', 'Etr', 'Pbr', 'Stu', 'Oro', 'Tor', 'Cdi', 'Cra', 'Ore', 'Ata', 'Cfu', 'Cgr', 'Pcl', 'Pos', 'Pse', 'Cden', 'Aco', 'Dip', 'Csu', 'Mnu', 'Pco', 'Cwa', 'Pba', 'Dac', 'Lun', 'Nsi', 'Rst', 'Pst', 'Acn', 'Tno', 'Ccu', 'Pan', 'Gfa', 'Hta', 'Dsp', 'Psu', 'Cei', 'Ndi', 'Cda', 'Dro', 'Cha', 'Pac', 'Cau', 'Coc', 'Pte', 'Mpe', 'Pde', 'Plo', 'Dpu', 'Rte', 'Fja', 'Hak', 'Mhe', 'Dno'),
  "cluster_2" = c('Oau', 'Ram', 'Tle', 'Tni', 'Edu')
)


sc_sp <- as.data.frame(sc_sp)
sc_sp <- sc_sp |> 
  mutate(spec_cluster = case_when(
    row.names(sc_sp) %in% unlist(spec_clusters[["cluster_0"]]) ~ "cluster_0",
    row.names(sc_sp) %in% unlist(spec_clusters[["cluster_1"]]) ~ "cluster_1",
    row.names(sc_sp) %in% unlist(spec_clusters[["cluster_2"]]) ~ "cluster_2",
    TRUE ~ "Other"  # This line is optional, it assigns "Other" to any species not found in the clusters
  )) |> 
  mutate(spec_cluster = as.factor(spec_cluster))

print(sc_sp)
sc_bp <- scores(spe.rda, display = "bp", choices = c(1, 2, 3), scaling = scaling)
plot_3d_rda(sc_sp, sc_bp, palette, perc, clusterdef=sc_sp$spec_cluster)

# code for exporting
n_rda <- 9
rda_scores <- 
  scores(spe.rda, display = "species", choices = 1:n_rda, scaling = 1) |> 
  as.data.frame()
weights <- round(100 * (summary(spe.rda)$cont$importance[2, 1:n_rda]), 2)
combined <- bind_rows(rda_scores, weights)
write.csv(combined, file="coastal.csv")

# no_cluster <- abio_std[["Coastal"]][2:ncol(abio_std[["Coastal"]])]
# mod0 <- rda(spec.hel[["Coastal"]] ~ 1, data=no_cluster)
# mod1 <- rda(spec.hel[["Coastal"]] ~ ., data=no_cluster)
# step.forward <-
#   ordiR2step(mod0,
#              scope=formula(mod1),
#              direction="forward",
#              prumtations=how(nperm=499))
# RsquareAdj(step.forward)


# 


## Global test of the RDA result
anova(spe1.rda, permutations = how(nperm = 999))
## Tests of all canonical axes

desired_significance <- 0.001
n_signif_axes <- function(group_name, scaling=1) {
  spe.rda <- rda(spec.hel[[group_name]] ~ ., abio_std[[group_name]])
  print(summary(spe.rda))
  signif <- anova(spe.rda, by = "axis", permutations = how(nperm = 1000))
  return(sum(signif$`Pr(>F)` <= desired_significance, na.rm=TRUE))
}


for (name in names(clusters)) {
  print(name)
  n_rda <-  n_signif_axes(name)
  rda_scores <- 
    scores(spe.rda, display = "species", choices = 1:n_rda, scaling = 1) |> 
    as.data.frame()
  weights <- round(100 * (summary(spe.rda)$cont$importance[2, 1:n_rda]), 2)
  combined <- bind_rows(rda_scores, weights)
  write.csv(combined, file=paste0(name, "_rdatest.csv"))
  print(paste0(name, " exported"))
}


n_signif_axes
# 
# 
# dev.new(width=7, height=7)
# triplot.rda(spe.rda,
#             site.sc = "lc",
#             scaling = 1,
#             cex.char2 = 0.7,
#             pos.env = 3,
#             plot.sites=FALSE,
#             label.sites=FALSE,
#             pos.centr = 1,
#             mult.arrow = 1.1,
#             mar.percent = 0.05,
#             select.spe = sel.sp
# )

