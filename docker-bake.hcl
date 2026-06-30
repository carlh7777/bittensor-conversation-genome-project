target "miner" {
  context = "."
  dockerfile = "Dockerfile"

  platforms = ["linux/amd64", "linux/arm64"]
  tags = [
    "readyai/bittensor-readyai-sn33:2.33.69",
  ]

  output = ["type=registry"]
}
