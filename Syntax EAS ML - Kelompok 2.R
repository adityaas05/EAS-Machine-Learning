#-----Import Packages-----
# Data Wrangling
library(tidyverse)
library(rwhatsapp)
library(wordcloud)
library(ggplot2)

# Text Preprocessing
library(textclean)
library(stringr)
library(tm)
library(reticulate)
library(dplyr)

# Cross Validation
library(rsample)

# Machine Learning
## Naive Bayes
library(e1071)
## LSTM
library(keras)

# Machine Learning Evaluation
library(caret)

#-----Import Dataset-----
gold <- read.csv("C:/Users/inet2/Downloads/gold_dataset.csv")
gold <- gold[!(gold$Price.Sentiment %in% c('neutral', 'none')), ]
head(gold, 3)

#-----Karakteristik Data-----
gold_new <- gold %>% 
  select(News, Price.Sentiment) %>% 
  mutate(Price.Sentiment = as.factor(Price.Sentiment))

glimpse(gold_new)

gold$Price.Sentiment %>% 
  table() %>% 
  prop.table()
sentiment_prop <- prop.table(table(gold$Price.Sentiment))
sentiment_df <- as.data.frame(sentiment_prop)
colnames(sentiment_df) <- c("Sentiment", "Proportion")
ggplot(sentiment_df, aes(x = Sentiment, y = Proportion, fill = Sentiment)) +
  geom_bar(stat = "identity") +
  labs(
    title = "Distribusi Sentimen Harga Emas",
    x = "Sentimen",
    y = "Proporsi"
  ) +
  theme_minimal()

#-----Data Teks Pre-processing-----
# Membuat sebuah object yang berisikan pattern yang biasanya ada pada sebuah link URL
url_pattern <- "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"

# Membuat object baru yang berisikan data dari object gold_new
check_url <- gold_new

# Menarik semua link URL yang ada dengan menggunakan fungsi di bawah ini 
check_url$ContentURL <- str_extract(check_url$News, url_pattern)
head(check_url)

unique(check_url$ContentURL)

# Implementasi dari fungsi lookup_emoji
check_emoticon <- rwhatsapp::lookup_emoji(gold_new, text_field = "News") 

head(check_emoticon)

unique(check_emoticon$emoji)

unique(check_emoticon$emoji_name)

teks_clean <- gold_new %>% 
  mutate(
    News_Clean = News %>% 
      str_to_lower() %>% 
      replace_internet_slang(replacement = paste0('{{ ', lexicon::hash_internet_slang[[2]], ' }}')) %>% 
      replace_contraction() %>% 
      replace_word_elongation() %>% 
      str_replace_all("[[:punct:]]", " ") %>% 
      str_replace_all("[[:digit:]]", " ") %>% 
      replace_white() %>% 
      str_squish()
  )

teks_clean %>% 
  select(News, News_Clean) %>% 
  head(5)

teks_clean <- gold_new %>% 
  mutate(
    News_Clean = News %>% 
      str_to_lower() %>% 
      replace_internet_slang(replacement = paste0('{{ ', lexicon::hash_internet_slang[[2]], ' }}')) %>% 
      replace_contraction() %>% 
      replace_word_elongation() %>% 
      replace_number(remove = T) %>% 
      str_replace_all("[[:punct:]]", " ") %>% 
      str_replace_all("[[:digit:]]", " ") %>% 
      str_replace_all("\\$", "") %>% 
      replace_white() %>% 
      str_squish()
  )

teks_clean %>% 
  select(News, News_Clean) %>% 
  head(5)

paste(teks_clean$News_Clean, collapse = " ") %>% 
  str_split(" ") %>% 
  unlist() %>% 
  n_distinct()

num_words = 1500
tokenizer <- text_tokenizer(num_words = num_words) %>% 
  fit_text_tokenizer(teks_clean$News_Clean)

Sys.setenv(TF_ENABLE_ONEDNN_OPTS = "0")

maxlen <- max(str_count(teks_clean$News_Clean, "\\w+")) + 1 
maxlen

data <- teks_clean %>% 
  mutate(label = factor(Price.Sentiment, levels = c("negative", "positive")),
         label = as.numeric(label),
         label = label - 1) %>%
  select(News_Clean, label) %>%
  na.omit()
head(data, 5)

#-----Split Data-----
set.seed(82)

intrain <- initial_split(data = data, prop = 0.8, strata = "label")

data_train <- training(intrain)
data_test <- testing(intrain)

set.seed(82)
inval <- initial_split(data = data_test, prop = 0.5, strata = "label")

data_val <- training(inval)
data_test <- testing(inval)

# prepare x
data_train_x <- texts_to_sequences(tokenizer, data_train$News_Clean) %>%
  pad_sequences(maxlen = maxlen, padding = "pre", truncating = "post")

data_val_x <- texts_to_sequences(tokenizer, data_val$News_Clean) %>%
  pad_sequences(maxlen = maxlen, padding = "pre", truncating = "post")

data_test_x <- texts_to_sequences(tokenizer, data_test$News_Clean) %>%
  pad_sequences(maxlen = maxlen, padding = "pre", truncating = "post")

# prepare y
data_train_y <- to_categorical(data_train$label, num_classes = 2)
data_val_y <- to_categorical(data_val$label, num_classes = 2)
data_test_y <- to_categorical(data_test$label, num_classes = 2)

#-----Model LSTM-----
model <- keras_model_sequential() %>%
  # layer input
  layer_embedding(
    name = "input",
    input_dim = num_words,
    input_length = maxlen,
    output_dim = 32
  ) %>%
  # layer lstm 
  layer_lstm(
    name = "lstm",
    units = 64,
    dropout = 0.5,
    return_sequences = FALSE, 
    kernel_initializer = initializer_random_uniform(minval = -0.05, maxval = 0.05)
  ) %>%
  # layer output
  layer_dense(
    name = "output",
    units = 2,
    activation = "softmax"
  )

model %>% compile(
  optimizer = optimizer_adam(learning_rate = 0.001),
  metrics = "accuracy",
  loss = "categorical_crossentropy"
)

summary(model)

model_lstm <- model %>% fit(
  data_train_x, data_train_y,
  batch_size = 256, 
  epochs = 10,
  verbose = 1,
  validation_data = list(
    data_val_x, data_val_y
  )
)

# model_lstm plot
plot(model_lstm)

#-----Evaluasi Model-----
test_pred <- model %>%
  predict(data_test_x) %>%
  k_argmax()

test_pred_vector <- as.array(test_pred)
eval_lstm <- confusionMatrix(
  factor(test_pred_vector,labels = c("negative", "positive")),
  factor(data_test$label,labels = c("negative", "positive"))
)
eval_lstm

# Dapatkan confusion Matrix dari hasil evaluasi LSTM
conf_matrix_lstm <- eval_lstm$table
print(conf_matrix_lstm)

# Buat heatmap dari confusion Matrix
heatmap_plot_lstm <- ggplot(data = as.data.frame(conf_matrix_lstm), aes(x = Reference, y = factor(Prediction, levels = c("negative", "positive")), fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), vjust = 1, size = 10) +
  scale_fill_gradient(low = "white", high = "blue") +
  labs(title = "Confusion Matrix Heatmap LSTM", x = "Reference", y = "Prediction") +
  theme_minimal() +
  theme(axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12))
print(heatmap_plot_lstm)

library(pROC)
# Membuat objek ROC dari prediksi dan label
roc_curve <- roc(as.numeric(data_test$label == "1"), as.numeric(test_pred_vector == "1"))
plot(roc_curve, main = "Kurva ROC", col = "blue", lwd = 2)
auc_value <- auc(roc_curve)
text(0.8, 0.2, paste("AUC =", round(auc_value, 3)), col = "blue")

