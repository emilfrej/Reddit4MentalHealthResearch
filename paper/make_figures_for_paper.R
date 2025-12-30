# =============================================================================
# make_figures_for_paper.R
# Generates all figures and tables for the paper
# =============================================================================
#
# REQUIRED FILES (relative to paper/ directory):
#
#   paper/
#   ├── meta_data_1880921.csv          <- from zst_to_csv.py
#   ├── predictor_values.csv           <- from log_reg.py
#   ├── top_bottom_topic_words-2.csv   <- from log_reg.py
#   ├── evaluations/                   <- from S3_hyperparamtuning.py
#   │   └── topic_quality_n*_seed*.csv
#   └── fit_time_testing/              <- from fit_time_testing.py
#       └── fit_times_*.csv
#
# OUTPUT FILES (saved to paper/figures/ directory):
#
#   paper/figures/
#   ├── post_counts.png
#   ├── gini.png
#   ├── min_posts_table.html
#   ├── top_predictors.png
#   ├── top_10_keyword_table.html
#   ├── S3_eval.png
#   ├── pred_fit_times_table.html
#   └── fit_time_testing.png
#
# USAGE:
#   cd paper/
#   Rscript make_figures_for_paper.R
#
# =============================================================================

# Load packages
if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, pracma, gt, ggrepel)

# Set working directory to script location if run interactively
# (Rscript already sets wd to script location)

cat("=== Making figures for paper ===\n\n")

# Create output directory if it doesn't exist
if (!dir.exists("../figures")) {

dir.create("../figures", recursive = TRUE)
}

# =============================================================================
# 1. Post counts bar chart
# =============================================================================
cat("1. Loading meta_data and creating post counts chart...\n")

meta_data <- read_csv("meta_data_1880921.csv", show_col_types = FALSE)

meta_data <- meta_data %>%
mutate(subreddit = paste0("r/", subreddit))

subreddit_counts <- meta_data %>%
group_by(subreddit) %>%
summarize(count = n(), .groups = "drop") %>%
mutate(subreddit = fct_reorder(subreddit, count))

ggplot(subreddit_counts, aes(x = reorder(subreddit, -count), y = count, fill = count)) +
geom_col() +
scale_fill_viridis_c(option = "C", end = 0.95) +
coord_flip() +
theme_classic() +
xlab("") +
ylab("#Posts") +
scale_y_continuous(labels = scales::comma) +
theme(legend.position = "none")

ggsave("figures/post_counts.png")
cat("   Saved: figures/post_counts.png\n")

# =============================================================================
# 2. Lorenz curve and Gini coefficient
# =============================================================================
cat("2. Creating Lorenz curve (Gini coefficient)...\n")

user_post_counts <- meta_data %>%
filter(author != -1) %>%  # -1 is deleted users
group_by(author) %>%
summarise(post_count = n(), .groups = "drop") %>%
arrange(post_count) %>%
mutate(
  cum_users = cumsum(rep(1, n())) / n(),
  cum_posts = cumsum(post_count) / sum(post_count)
)

ggplot(user_post_counts, aes(x = cum_users, y = cum_posts)) +
geom_line(color = "orange") +
geom_abline(slope = 1, linetype = "dashed") +
labs(
  x = "Cumulative fraction of users",
  y = "Cumulative fraction of posts"
) +
coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
theme_minimal()

ggsave("figures/gini.png")

# Calculate Gini coefficient
area <- trapz(user_post_counts$cum_users, user_post_counts$cum_posts)
gini <- 1 - 2 * area
cat("   Gini coefficient:", round(gini, 3), "\n")
cat("   Saved: figures/gini.png\n")

# =============================================================================
# 3. Minimum post counts table
# =============================================================================
cat("3. Creating minimum post counts table...\n")

min_post_counts <- user_post_counts %>%
summarize(
  "1" = sum(post_count >= 1),
  "5" = sum(post_count >= 5),
  "10" = sum(post_count >= 10),
  "25" = sum(post_count >= 25),
  "50" = sum(post_count >= 50),
  "100" = sum(post_count >= 100)
) %>%
pivot_longer(everything(), names_to = "min_posts", values_to = "n_users")

min_post_counts %>%
gt() %>%
cols_label(
  min_posts = "Minimum Posts",
  n_users = "Number of Users"
) %>%
tab_options(
  table.font.size = px(14),
  data_row.padding = px(6)
) %>%
gtsave("figures/min_posts_table.html")

cat("   Saved: figures/min_posts_table.html\n")

# =============================================================================
# 4. Top predictors plot
# =============================================================================
cat("4. Creating top predictors plot...\n")

predictor_vals <- read_csv("predictor_values.csv", show_col_types = FALSE) %>%
arrange(desc(abs_estimate)) %>%
head(300)

ggplot(predictor_vals, aes(x = reorder(topic, abs_estimate), y = estimate, fill = estimate)) +
geom_col() +
scale_fill_viridis_c(option = "C", end = 0.95) +
coord_flip() +
theme_classic() +
xlab("Topics") +
theme(
  axis.text.y = element_blank(),
  axis.ticks.y = element_blank()
) +
ylab("Estimated Coefficient (log-odds)") +
scale_y_continuous(labels = scales::comma) +
theme(legend.position = "none")

ggsave("figures/top_predictors.png")
cat("   Saved: figures/top_predictors.png\n")

# =============================================================================
# 5. Top-k keyword table
# =============================================================================
cat("5. Creating top-k keyword table...\n")

top_bottom_words <- read_csv("top_bottom_topic_words-2.csv", show_col_types = FALSE)

make_topk_keyword_table <- function(
predictor_vals,
top_bottom_words,
k = 20,
output_folder = NULL
) {
# Helper: parse python-style lists
parse_word_list <- function(x) {
  x %>%
    stringr::str_remove_all("^\\[|\\]$") %>%
    stringr::str_remove_all("'") %>%
    stringr::str_split(",\\s*") %>%
    unlist()
}

# Select top-k predictors
predictor_topk <- predictor_vals %>%
  arrange(desc(abs_estimate)) %>%
  slice_head(n = k)

topics <- predictor_topk$topic

# Subset keyword table to top-k topics
tbw <- top_bottom_words %>%
  select(1, all_of(as.character(topics)))

# Extract top / bottom words
top_words <- tbw[1, -1] |> unlist(use.names = FALSE)
bottom_words <- tbw[2, -1] |> unlist(use.names = FALSE)

pretty_format_words <- tibble(
  topic = topics,
  top_words = top_words,
  bottom_words = bottom_words,
  sign = if_else(predictor_topk$estimate > 0, "+", "-")
) %>%
  mutate(
    top_words = map(top_words, parse_word_list),
    bottom_words = map(bottom_words, parse_word_list),
    top_words = map_chr(top_words, ~ paste(.x, collapse = ", ")),
    bottom_words = map_chr(bottom_words, ~ paste(.x, collapse = ", "))
  )

gt_table <- pretty_format_words %>%
  gt() %>%
  cols_label(
    topic = "Topic",
    top_words = "Positive Keywords",
    bottom_words = "Negative Keywords",
    sign = "Sign of Coefficient"
  ) %>%
  cols_width(
    topic ~ px(80),
    top_words ~ px(350),
    bottom_words ~ px(350),
    sign ~ px(50)
  ) %>%
  tab_options(
    table.font.size = px(14),
    data_row.padding = px(6)
  )

if (!is.null(output_folder)) {
  output_html <- paste0(output_folder, "/top_", k, "_keyword_table.html")
  gtsave(gt_table, output_html)
}

invisible(gt_table)
}

make_topk_keyword_table(
predictor_vals = predictor_vals,
top_bottom_words = top_bottom_words,
k = 10,
output_folder = "../figures"
)
cat("   Saved: figures/top_10_keyword_table.html\n")

# =============================================================================
# 6. S3 evaluation plot (coherence vs topics)
# =============================================================================
cat("6. Creating S3 evaluation plot...\n")

csv_files <- list.files(
path = "evaluations",
pattern = "\\.csv$",
full.names = TRUE
)

combined_df <- csv_files %>%
map_dfr(~ read_csv(.x, show_col_types = FALSE))

eval_summary <- combined_df %>%
mutate(c_v = sqrt(c_in * c_ex)) %>%
group_by(n_topics) %>%
summarize(
  diversity_mean = mean(diversity),
  c_v_mean = mean(c_v),
  c_v_sd = sd(c_v),
  .groups = "drop"
)

ggplot(eval_summary, aes(x = n_topics, y = c_v_mean, color = diversity_mean)) +
geom_point() +
geom_errorbar(
  aes(ymin = c_v_mean - c_v_sd, ymax = c_v_mean + c_v_sd),
  width = 0.1
) +
scale_color_viridis_c(option = "C", end = 0.95) +
labs(color = "Mean Diversity", x = "Topics", y = "Mean Coherence") +
theme_classic()

ggsave("figures/S3_eval.png")
cat("   Saved: figures/S3_eval.png\n")

# =============================================================================
# 7. Fit time testing plot
# =============================================================================
cat("7. Creating fit time testing plot...\n")

csv_files <- list.files(
  path = "fit_time_testing",
  pattern = "\\.csv$",
  full.names = TRUE
)

fit_times <- csv_files %>%
  map_dfr(read_csv, id = "source") %>% 
  #get number of topics from filepath
  mutate(corpus_size = as.numeric(str_extract(source, "\\d+(?=\\.csv$)")),
         time = time / 60)

n_docs <- 1880921

#fit a log-log linear model and predict fitting time for each model. round result in hours
predictions <- fit_times %>%
  group_by(model_name) %>%
  nest() %>%
  #fit a model for each
  mutate(
    fit = map(data, ~ lm(log(time) ~ log(corpus_size), data = .x)),
    #predict, convert to hours, then round
    pred_time = map_dbl(
      fit,
      ~ round(exp(predict(.x, newdata = data.frame(corpus_size = n_docs))) / 60, 2)
    )
  ) %>%
  #keep rel cols
  select(model_name, pred_time) %>% 
  #round for gttable not getting confused
  ungroup()



#make pretty table of prediticed times
pred_fit_times_table <- predictions %>%
  gt() %>%
  cols_label(
    model_name = "Model",
    pred_time = "Predicted Time (hours)",
  ) %>%
  tab_options(
    table.font.size = px(14),
    data_row.padding = px(6)
  )
gtsave(pred_fit_times_table, "../figures/pred_fit_times_table.html")

#add pred fit times for coloring later
fit_times_plot <- fit_times %>%
  left_join(predictions, by = "model_name")

#make df of labes for annotating
label_df <- fit_times_plot %>%
  group_by(model_name) %>%
  slice_max(corpus_size, n = 1) %>%
  ungroup()



#plot fit times, annontate, and color according to fit time
fit_times_plot <- ggplot(
  fit_times_plot,
  aes(
    x = corpus_size,
    y = time,
    color = pred_time,
    group = model_name
  )
) +
  geom_point(alpha = 0.5) +
  geom_line(alpha = 0.6) +
  geom_text_repel(
    data = label_df,
    aes(label = model_name),
    hjust = 0,
    direction = "y",
    nudge_x = 0.05 * max(fit_times_plot$corpus_size),
    box.padding = 0.6,
    point.padding = 0.4,
    force = 2,
    force_pull = 0,
    segment.color = "grey70",
    segment.size = 0.3,
    show.legend = FALSE
  ) +
  scale_color_viridis_c(
    option = "C",
    direction = -1,
    end = 0.92
  ) +
  theme_classic() +
  labs(
    x = "# Documents",
    y = "Minutes",
    color = "Predicted Fit Time (hours)"
  ) +
  theme(legend.position = "none") +
  coord_cartesian(clip = "off")
ggsave("figures/fit_time_testing.png")
###acutal time for 10 topic S3 15940.465493679047 (seconds) 265 minutes ##



# =============================================================================
cat("\n=== All figures generated successfully! ===\n")
