
library(plotly)
library(RColorBrewer)

plot_3d_rda <- function(species_scores, abio_scores, color_palette, perc, clusterdef=NULL) {
  # Ensure that 'clusterdef' is a factor
  if (!is.null(clusterdef)) {
    clusterdef <- as.factor(clusterdef)
  }
  
  p <- plot_ly()
  if (!is.null(clusterdef)) {
    p <- p |>
      # Add species points
      add_trace(
        type = "scatter3d",
        mode = "markers",
        x = species_scores[, 1],
        y = species_scores[, 2],
        z = species_scores[, 3],
        text = rownames(species_scores),
        textposition = "top center",
        marker = list(symbol = "circle", size = 10),
        color = ~clusterdef,
        colors = color_palette, # Use the provided color palette
        name = ~clusterdef,
        showlegend = TRUE
      )
  } else {
    p <- p |> add_trace(
      type = "scatter3d",
      mode = "markers",
      x = species_scores[, 1],
      y = species_scores[, 2],
      z = species_scores[, 3],
      text = rownames(species_scores),
      textposition = "top center",
      marker = list(symbol = "circle", size = 10),
      name = "Species",
      showlegend = TRUE
    )
  }
  # Create the plot
  
  
  # Add arrows for explanatory variables
  for (i in 1:nrow(abio_scores)) {
    p <- p |>
      add_trace(
        type = "scatter3d",
        mode = "lines+markers+text",
        x = c(0, abio_scores[i, 1]),
        y = c(0, abio_scores[i, 2]),
        z = c(0, abio_scores[i, 3]),
        line = list(color = "red", width = 3),
        marker = list(size = 2, color = "red", symbol = "triangle"),
        text = rownames(abio_scores)[i],
        textposition = "top center",
        name = rownames(abio_scores)[i],
        showlegend = FALSE
      )
  }
  
  # Customize layout
  p <- p |>
    layout(
      title = "Triplot RDA - scaling 2 for Hansweert",
      scene = list(
        xaxis = list(title = paste0("RDA1 (", perc[1], "%)")),
        yaxis = list(title = paste0("RDA2 (", perc[2], "%)")),
        zaxis = list(title = paste0("RDA3 (", perc[3], "%)"))
      ),
      showlegend = TRUE
    )
  
  return(p)
}

plot_2d_rda <- function(species_scores, abio_scores, color_palette, perc, clusterdef=NULL) {
  # Ensure that 'clusterdef' is a factor
  if (!is.null(clusterdef)) {
    clusterdef <- as.factor(clusterdef)
  }
  
  p <- plot_ly()
  if (!is.null(clusterdef)) {
    p <- p |>
      # Add species points
      add_trace(
        type = "scatter",
        mode = "markers",
        x = species_scores[, 1],
        y = species_scores[, 2],
        text = rownames(species_scores),
        textposition = "top center",
        marker = list(symbol = "circle", size = 10),
        color = ~clusterdef,
        colors = color_palette, # Use the provided color palette
        name = ~clusterdef,
        showlegend = TRUE
      )
  } else {
    p <- p |> add_trace(
      type = "scatter",
      mode = "markers",
      x = species_scores[, 1],
      y = species_scores[, 2],
      text = rownames(species_scores),
      textposition = "top center",
      marker = list(symbol = "circle", size = 10),
      name = "Species",
      showlegend = TRUE
    )
  }
  # Create the plot
  
  
  # Add arrows for explanatory variables
  for (i in 1:nrow(abio_scores)) {
    p <- p |>
      add_trace(
        type = "scatter",
        mode = "lines+markers+text",
        x = c(0, abio_scores[i, 1]),
        y = c(0, abio_scores[i, 2]),
        line = list(color = "red", width = 3),
        marker = list(size = 2, color = "red", symbol = "triangle"),
        text = rownames(abio_scores)[i],
        textposition = "top center",
        name = rownames(abio_scores)[i],
        showlegend = FALSE
      )
  }
  
  # Customize layout
  p <- p |>
    layout(
      title = "Biplot RDA - scaling 2",
      xaxis = list(title = paste0("RDA1 (", perc[1], "%)")),
      yaxis = list(title = paste0("RDA2 (", perc[2], "%)")),
      showlegend = TRUE
    )
  
  return(p)
}


