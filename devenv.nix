{ pkgs, ... }: 

{
  languages.python.enable = true;
  languages.python.version = "3.11.8";
  languages.python.poetry.enable = true;
  languages.python.poetry.activate.enable = true;
}
