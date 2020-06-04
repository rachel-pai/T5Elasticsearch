CREATE DATABASE IF NOT EXISTS `docdb`;
USE `docdb`;
CREATE TABLE IF NOT EXISTS `docs` (
    `doc_id` INT AUTO_INCREMENT PRIMARY KEY,
    `title` VARCHAR(255) NOT NULL,
    `start_date` DATE,
    `description` TEXT,
    `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
INSERT INTO `docs`
  (title, start_date,description)
VALUES
  ("haha","2020-08-09","this is description"),
  ("haqye","2020-08-08","this is the second desc"),
  ("haqye","2020-08-08","this is the thjir desc"),
  ("haqye","2020-08-08","rachel is very smart"),
  ("haqye","2020-08-08","you are brialant!"),
  ("esa","2020-08-08","i love you very much"),
  ("aqx","2020-08-08","this is what i want"),
  ("xxa","2020-08-08","do not do this!"),
  ("dez","2020-08-08","i like you"),
  ("zzq","2020-08-08","i hare you"),
  ("veaz","2020-08-08","happy birthday"),
  ("cewdx","2020-08-08","this is the third desc"),
  ("xxqq","2020-08-08","this is the fourth desc"),
  ("feax","2020-08-08","this is the fifth desc"),
  ("rwqds","2020-08-08","this is the fixth desc"),
  ("cdewa","2020-08-08","this is the seventh desc");
