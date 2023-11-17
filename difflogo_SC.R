library(DiffLogo)
library(MotifDb)
library(plyr)
library(stringi)

parse_near_atoms <- function(data_string) {
  # Split the string into key-value pairs
  # Assuming your data is a character vector containing the Python dictionary string
  
  # Extract matches using regex
  matches <- stri_match_all_regex(data_string, "\\((.*?)\\):\\s*([^,]+),?")
  
  # Extract keys and values
  keys <- matches[[1]][, 2]
  values <- as.numeric(matches[[1]][, 3])
  
  # Create a data frame
  your_data_frame <- data.frame(key = keys, value = values)
  
  # Split the "key" column into two columns
  your_data_frame[c("column1", "column2")] <- stri_split_fixed(your_data_frame$key, ",", simplify = TRUE)
  
  # Remove the original "key" column
  your_data_frame <- your_data_frame[, c("column1", "column2", "value")]
  
  your_data_frame$column2 <- gsub("[\ ]", "", your_data_frame$column2)
  your_data_frame$column2 <- gsub("[\']", "\"", your_data_frame$column2)
  your_data_frame$column2 <- gsub("\"\"", "\'", your_data_frame$column2)
  your_data_frame$column2 <- gsub("\"", "", your_data_frame$column2)
  # your_data_frame$column2 <- as.character(your_data_frame$column2)
  
  # Map numerical values to letters in "column1"
  your_data_frame$column1 <- ifelse(your_data_frame$column1 == "0", "A",
                                    ifelse(your_data_frame$column1 == "1", "C",
                                           ifelse(your_data_frame$column1 == "2", "G",
                                                  ifelse(your_data_frame$column1 == "3", "U", NA))))
  
  return(your_data_frame)
}

plot_motifdiff <- function(motif_array) {
  heatmap(motif_array, 
          Rowv = NA, 
          Colv = NA, 
          # scale = "column",
          main = "Heatmap of Motif Matrix",
          col = heat.colors(256))
  
  legend_values <- seq(min(motif_array),
      max(motif_array),
      length.out = 10)
  
  rounded_legend_values <- round(legend_values, 3)
  legend("bottomleft", 
         legend = rounded_legend_values,  # Modify as needed
         fill = colorRampPalette(c("red", "yellow", "white"))(11), 
         title = "Legend",
         cex = 0.8)
  print(sum(motif_array))
}

create_heatmap <- function(matrix_data, legend_values, legend_title) {
  heatmap(matrix_data,
          Rowv = NA,
          Colv = NA,
          scale = "column",
          main = "Heatmap of Motif Matrix",
          xlab = "Columns",
          ylab = "Rows",
          col = heat.colors(256))
  
  # Add the legend
  legend("bottomleft",
         legend = legend_values,
         fill = colorRampPalette(c("red", "yellow", "white"))(length(legend_values)),
         title = legend_title,
         cex = 0.8)
}

heavyatomlist = c("C1'",'C2',"C2'","C3'",'C4',"C4'",'C5',
                  "C5'",'C6','C8','N1','N2','N3','N4','N6','N7',
                  'N9','O2',"O2'","O3'",'O4',"O4'","O5'",'O6',
                  'OP1','OP2','OP3','P')

parsed_file <- read.csv("~/Downloads/6W3M_H1'")
motif_list <- list()
for (i in 1:nrow(parsed_file)) {
  data_string  <- parsed_file[i,2]
  df <- parse_near_atoms(data_string)
  motif_df <- matrix(0, nrow=4, ncol=length(heavyatomlist))
  rownames(motif_df) <- c("A", "C", "G", "U")
  for (j in 1:length(heavyatomlist)) {
    rows <- which(df$column2 == heavyatomlist[j])
    if(length(rows) > 0) {
      for (k in 1:length(rows)) {
        motif_df[which(rownames(motif_df)==df[rows[k],1]),j] <-
          motif_df[which(rownames(motif_df)==df[rows[k],1]),j] + df[rows[k],3]
      } 
    }
  }
  norm_motif_df <- apply(motif_df, 2, function(x) {
    if (sum(x) == 0) {
      return(rep(0.25,4))
    } else {
      # return((x - min(x)) / (max(x) - min(x)))
      return(x/sum(x))
    }
  })
  norm_motif_df[is.nan(norm_motif_df)] <- 0.25
  motif_list[[i]] <- norm_motif_df
}
names(motif_list) <- parsed_file$atom

# # examples that perform well
# seqLogo::seqLogo(pwm = motif_list[[35]])
# seqLogo::seqLogo(pwm = motif_list[[20]])
# seqLogo::seqLogo(pwm = motif_list[[33]])
# diffLogoFromPwm(motif_list[[35]], motif_list[[20]])
# diffLogoFromPwm(motif_list[[35]], motif_list[[33]])
# 
# # examples that perform badly
# seqLogo::seqLogo(pwm = motif_list[[4]])
# seqLogo::seqLogo(pwm = motif_list[[2]])
# seqLogo::seqLogo(pwm = motif_list[[1]])
# diffLogoFromPwm(motif_list[[4]], motif_list[[2]])
# diffLogoFromPwm(motif_list[[4]], motif_list[[1]])

diffLogoTable(PWMs = motif_list[c(35,20,2,1)],
                    configuration = list(showSequenceLogosTop = F))

plot_motifdiff(differenceOfICs(motif_list[[35]], motif_list[[20]]))
plot_motifdiff(differenceOfICs(motif_list[[35]], motif_list[[33]]))
plot_motifdiff(differenceOfICs(motif_list[[35]], motif_list[[4]]))
plot_motifdiff(differenceOfICs(motif_list[[35]], motif_list[[2]]))
plot_motifdiff(differenceOfICs(motif_list[[35]], motif_list[[1]]))

# Define legend values and title
legend_values <- seq(-0.1, 0.05, length.out = 11)  # Modify as needed
rounded_legend_values <- round(legend_values, 2)  # Adjust the number of decimal places
legend_title <- "Legend"

# Create and display the first heatmap
create_heatmap(differenceOfICs(motif_list[[35]], motif_list[[20]]), rounded_legend_values, legend_title)
create_heatmap(differenceOfICs(motif_list[[35]], motif_list[[33]]), rounded_legend_values, legend_title)
create_heatmap(differenceOfICs(motif_list[[35]], motif_list[[4]]), rounded_legend_values, legend_title)
create_heatmap(differenceOfICs(motif_list[[35]], motif_list[[2]]), rounded_legend_values, legend_title)
create_heatmap(differenceOfICs(motif_list[[35]], motif_list[[1]]), rounded_legend_values, legend_title)

parsed_file <- read.csv("~/Downloads/6HYK_H1'")
motif_list <- list()
for (i in 1:nrow(parsed_file)) {
  data_string  <- parsed_file[i,2]
  df <- parse_near_atoms(data_string)
  motif_df <- matrix(0, nrow=4, ncol=length(heavyatomlist))
  rownames(motif_df) <- c("A", "C", "G", "U")
  for (j in 1:length(heavyatomlist)) {
    rows <- which(df$column2 == heavyatomlist[j])
    if(length(rows) > 0) {
      for (k in 1:length(rows)) {
        motif_df[which(rownames(motif_df)==df[rows[k],1]),j] <-
          motif_df[which(rownames(motif_df)==df[rows[k],1]),j] + df[rows[k],3]
      } 
    }
  }
  norm_motif_df <- apply(motif_df, 2, function(x) {
    if (sum(x) == 0) {
      return(rep(0.25,4))
    } else {
      # return((x - min(x)) / (max(x) - min(x)))
      return(x/sum(x))
    }
  })
  norm_motif_df[is.nan(norm_motif_df)] <- 0.25
  motif_list[[i]] <- norm_motif_df
}
names(motif_list) <- parsed_file$atom

# # examples that perform well
# seqLogo::seqLogo(pwm = motif_list[[4]])
# seqLogo::seqLogo(pwm = motif_list[[2]])
# seqLogo::seqLogo(pwm = motif_list[[25]])
# diffLogoFromPwm(motif_list[[4]], motif_list[[2]])
# diffLogoFromPwm(motif_list[[4]], motif_list[[25]])
# 
# # examples that perform badly
# seqLogo::seqLogo(pwm = motif_list[[11]])
# seqLogo::seqLogo(pwm = motif_list[[7]])
# seqLogo::seqLogo(pwm = motif_list[[20]])
# diffLogoFromPwm(motif_list[[11]], motif_list[[7]])
# diffLogoFromPwm(motif_list[[7]], motif_list[[20]])

diffLogoTable(PWMs = motif_list[c(4,2,7,20)],
              configuration = list(showSequenceLogosTop = F))

  
plot_motifdiff(differenceOfICs(motif_list[[11]], motif_list[[7]]))
plot_motifdiff(differenceOfICs(motif_list[[7]], motif_list[[20]]))
plot_motifdiff(differenceOfICs(motif_list[[20]], motif_list[[4]]))
plot_motifdiff(differenceOfICs(motif_list[[20]], motif_list[[2]]))
plot_motifdiff(differenceOfICs(motif_list[[7]], motif_list[[4]]))

# Create and display the first heatmap
create_heatmap(differenceOfICs(motif_list[[4]], motif_list[[2]]), rounded_legend_values, legend_title)
create_heatmap(differenceOfICs(motif_list[[4]], motif_list[[25]]), rounded_legend_values, legend_title)
create_heatmap(differenceOfICs(motif_list[[4]], motif_list[[11]]), rounded_legend_values, legend_title)
create_heatmap(differenceOfICs(motif_list[[4]], motif_list[[7]]), rounded_legend_values, legend_title)
create_heatmap(differenceOfICs(motif_list[[4]], motif_list[[20]]), rounded_legend_values, legend_title)

pdf("output.pdf")
par(mar = c(2, 2, 2, 2))
diffLogoTable(PWMs = motif_list[c(4,2,25,11,7,20)])
dev.off()
  
