def run_pipeline(youtube_url: str, start_time: str, end_time: str, question: str | None = None):
	return {
		"youtube_url": youtube_url,
		"start_time": start_time,
		"end_time": end_time,
		"question": question,
		"status": "pipeline stub created"
	}
