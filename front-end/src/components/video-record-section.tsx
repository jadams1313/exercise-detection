import { useState } from 'react';
import { Button } from './ui/button';
import { Card } from './ui/card';
import { Video, Square, Play, Trash2, BarChart3 } from 'lucide-react';

export function VideoRecordSection() {
  const [isRecording, setIsRecording] = useState(false);
  const [hasRecording, setHasRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);

  const handleRecord = () => {
    if (!isRecording) {
      // Start recording logic would go here
      setIsRecording(true);
      console.log('Starting recording...');
      
      // Simulate recording timer (in real app, this would be actual recording time)
      const timer = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);
      
      // Auto-stop after 5 seconds for demo purposes
      setTimeout(() => {
        setIsRecording(false);
        setHasRecording(true);
        clearInterval(timer);
      }, 5000);
    } else {
      // Stop recording logic would go here
      setIsRecording(false);
      setHasRecording(true);
      console.log('Stopping recording...');
    }
  };

  const handleDelete = () => {
    setHasRecording(false);
    setRecordingTime(0);
    console.log('Recording deleted');
  };

  const handlePreview = () => {
    console.log('Playing preview...');
    // Preview logic would go here
  };

  const handleAnalyze = () => {
    console.log('Starting analysis...');
    // Analysis logic would go here
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <Card className="bg-card border-white/10 p-8">
      <div className="text-center space-y-6">
        <div className="w-16 h-16 bg-red-600/20 rounded-full flex items-center justify-center mx-auto">
          <Video className="w-8 h-8 text-red-400" />
        </div>
        
        <div>
          <h3 className="text-white mb-2">Record Your Workout</h3>
          <p className="text-muted-foreground">
            {hasRecording 
              ? 'Recording complete! Choose an action below to continue.'
              : 'Start recording to analyze your exercise form and technique in real-time'
            }
          </p>
        </div>

        <div className="bg-black/40 rounded-lg aspect-video w-full max-w-md mx-auto border border-white/10 flex items-center justify-center">
          {isRecording ? (
            <div className="text-center">
              <div className="w-4 h-4 bg-red-500 rounded-full animate-pulse mx-auto mb-2"></div>
              <p className="text-sm text-red-400">Recording... {formatTime(recordingTime)}</p>
            </div>
          ) : hasRecording ? (
            <div className="text-center">
              <Play className="w-12 h-12 text-green-400 mx-auto mb-2" />
              <p className="text-sm text-green-400">Recording Complete</p>
              <p className="text-xs text-white/60">{formatTime(recordingTime)} duration</p>
            </div>
          ) : (
            <div className="text-center">
              <Video className="w-12 h-12 text-white/20 mx-auto mb-2" />
              <p className="text-sm text-white/40">Camera Preview</p>
            </div>
          )}
        </div>

        {hasRecording ? (
          <div className="flex gap-3 justify-center flex-wrap">
            <Button
              onClick={handleDelete}
              variant="outline"
              className="border-red-500/50 text-red-400 hover:bg-red-500/10 hover:border-red-500"
            >
              <Trash2 className="w-4 h-4 mr-2" />
              Delete
            </Button>
            <Button
              onClick={handlePreview}
              variant="outline"
              className="border-white/20 text-white hover:bg-white/10"
            >
              <Play className="w-4 h-4 mr-2" />
              Preview
            </Button>
            <Button
              onClick={handleAnalyze}
              className="bg-red-600 hover:bg-red-700 text-white"
            >
              <BarChart3 className="w-4 h-4 mr-2" />
              Analyze
            </Button>
          </div>
        ) : (
          <div className="flex gap-3 justify-center">
            <Button
              onClick={handleRecord}
              disabled={isRecording}
              className="bg-red-600 hover:bg-red-700 text-white disabled:opacity-50"
            >
              {isRecording ? (
                <>
                  <Square className="w-4 h-4 mr-2" />
                  Recording...
                </>
              ) : (
                <>
                  <Video className="w-4 h-4 mr-2" />
                  Start Recording
                </>
              )}
            </Button>
          </div>
        )}
      </div>
    </Card>
  );
}