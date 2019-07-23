library(tidyverse)
library(reticulate)

np <- import("numpy")

dat <- tibble(
  sex = np$load("/Dedicated/jmichaelson-sdata/neuroimaging/processed/numpy/2019-07-22-processed_filtered_sex.npy") %>% unlist(),
  site = np$load("/Dedicated/jmichaelson-sdata/neuroimaging/processed/numpy/2019-07-22-processed_filtered_site.npy") %>% unlist(),
  study = np$load("/Dedicated/jmichaelson-sdata/neuroimaging/processed/numpy/2019-07-22-processed_filtered_study.npy") %>% unlist()
) 

dat_recode <- dat %>% 
  mutate(
    sex = case_when(
      as.numeric(sex) > 2 ~ 'U', 
      TRUE ~ sex), 
    sex =  substr(sex, 1, 1) %>% tolower(),
    sex = case_when(
      sex == 'f' ~ '1', 
      sex == 'm' ~ '0', 
      sex %in% c('0','1', '2') ~ sex, 
      TRUE ~ '0.5')
    ) %>%
group_by(study, site) %>%
  mutate(
    u = list(unique(sex)), 
    has_2 = "2" %in% unlist(u), 
    sex = case_when(
      has_2 ~ as.character(as.numeric(sex) - 1), 
      !has_2 ~ sex
    )
  )

np$save("/Dedicated/jmichaelson-sdata/neuroimaging/processed/numpy/2019-07-22-processed_filtered_sex_recode.npy", as.numeric(dat_recode$sex) )

