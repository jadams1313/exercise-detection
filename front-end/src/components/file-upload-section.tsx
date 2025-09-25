import { useState } from 'react';
import { Button } from './ui/button';
import { Card } from './ui/card';
import { Upload, FileVideo, CheckCircle, Trash2, Play, BarChart3, Plus } from 'lucide-react';

interface UploadedFile {
  name: string;
  size: string;
  duration: string;
}

export function FileUploadSection() {
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [isDragging, setIsDragging] = useState(false);

  const handleFileUpload = (files: FileList | null) => {
    if (files) {
      const newFiles = Array.from(files).map(file => ({
        name: file.name,
        size: (file.size / (1024 * 1024)).toFixed(1) + ' MB',
        duration: '2:30' // Mock duration - in real app this would be calculated
      }));
      setUploadedFiles(prev => [...prev, ...newFiles]);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    handleFileUpload(e.dataTransfer.files);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDelete = (index: number) => {
    setUploadedFiles(prev => prev.filter((_, i) => i !== index));
    console.log('File deleted');
  };

  const handlePreview = (fileName: string) => {
    console.log('Playing preview for:', fileName);
    // Preview logic would go here
  };

  const handleAnalyze = (fileName: string) => {
    console.log('Starting analysis for:', fileName);
    // Analysis logic would go here
  };

  return (
    <Card className="bg-card border-white/10 p-8">
      <div className="text-center space-y-6">
        <div className="w-16 h-16 bg-red-600/20 rounded-full flex items-center justify-center mx-auto">
          <Upload className="w-8 h-8 text-red-400" />
        </div>
        
        <div>
          <h3 className="text-white mb-2">Upload Exercise Videos</h3>
          <p className="text-muted-foreground">
            {uploadedFiles.length > 0 
              ? `${uploadedFiles.length} video${uploadedFiles.length === 1 ? '' : 's'} uploaded. Choose actions for each video below.`
              : 'Upload existing workout videos for detailed motion analysis and feedback'
            }
          </p>
        </div>

        {uploadedFiles.length === 0 ? (
          <div
            className={`border-2 border-dashed rounded-lg p-8 transition-colors ${
              isDragging
                ? 'border-red-400 bg-red-400/10'
                : 'border-white/20 hover:border-white/30'
            }`}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
          >
            <div className="text-center space-y-4">
              <Upload className={`w-12 h-12 mx-auto ${isDragging ? 'text-red-400' : 'text-white/40'}`} />
              <div>
                <p className="text-white mb-1">Drag and drop your videos here</p>
                <p className="text-sm text-muted-foreground">or click to browse files</p>
              </div>
              <input
                type="file"
                multiple
                accept="video/*"
                className="hidden"
                id="file-upload"
                onChange={(e) => handleFileUpload(e.target.files)}
              />
              <Button
                onClick={() => document.getElementById('file-upload')?.click()}
                variant="outline"
                className="border-white/20 text-white hover:bg-white/10"
              >
                <Upload className="w-4 h-4 mr-2" />
                Choose Files
              </Button>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="space-y-3">
              {uploadedFiles.map((file, index) => (
                <div
                  key={index}
                  className="p-4 bg-card/30 rounded-lg border border-white/10"
                >
                  <div className="flex items-center gap-3 mb-3">
                    <div className="w-10 h-10 bg-green-600/20 rounded-lg flex items-center justify-center">
                      <CheckCircle className="w-5 h-5 text-green-400" />
                    </div>
                    <div className="flex-1 text-left">
                      <div className="flex items-center gap-2">
                        <FileVideo className="w-4 h-4 text-white/60" />
                        <span className="text-white text-sm">{file.name}</span>
                      </div>
                      <p className="text-xs text-muted-foreground">
                        {file.size} • {file.duration}
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex gap-2 justify-center">
                    <Button
                      onClick={() => handleDelete(index)}
                      variant="outline"
                      size="sm"
                      className="border-red-500/50 text-red-400 hover:bg-red-500/10 hover:border-red-500"
                    >
                      <Trash2 className="w-3 h-3 mr-1" />
                      Delete
                    </Button>
                    <Button
                      onClick={() => handlePreview(file.name)}
                      variant="outline"
                      size="sm"
                      className="border-white/20 text-white hover:bg-white/10"
                    >
                      <Play className="w-3 h-3 mr-1" />
                      Preview
                    </Button>
                    <Button
                      onClick={() => handleAnalyze(file.name)}
                      size="sm"
                      className="bg-red-600 hover:bg-red-700 text-white"
                    >
                      <BarChart3 className="w-3 h-3 mr-1" />
                      Analyze
                    </Button>
                  </div>
                </div>
              ))}
            </div>
            
            <Button
              onClick={() => document.getElementById('file-upload')?.click()}
              variant="outline"
              className="border-white/20 text-white hover:bg-white/10 w-full"
            >
              <Plus className="w-4 h-4 mr-2" />
              Add More Videos
            </Button>
            <input
              type="file"
              multiple
              accept="video/*"
              className="hidden"
              id="file-upload"
              onChange={(e) => handleFileUpload(e.target.files)}
            />
          </div>
        )}
      </div>
    </Card>
  );
}