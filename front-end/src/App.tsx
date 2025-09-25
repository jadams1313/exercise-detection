import { Header } from './components/header';
import { VideoRecordSection } from './components/video-record-section';
import { FileUploadSection } from './components/file-upload-section';
import { FitnessJourneySection } from './components/fitness-journey-section';
import { FeaturesSection } from './components/features-section';
import { CardDeck } from './components/card-deck';
import { Button } from './components/ui/button';
import { ArrowRight, Play } from 'lucide-react';
import { ImageWithFallback } from './components/figma/ImageWithFallback';

export default function App() {
  return (
    <div className="min-h-screen bg-black text-white">
      <Header />
      
      {/* Hero Section */}
      <section className="relative py-20 px-4">
        <div className="absolute inset-0 bg-gradient-to-br from-red-600/20 via-transparent to-transparent"></div>
        <div className="container mx-auto text-center relative z-10">
          <div className="max-w-4xl mx-auto space-y-8">
            <div className="w-24 h-24 bg-red-600/20 rounded-full flex items-center justify-center mx-auto">
              <ImageWithFallback 
                src="https://images.unsplash.com/photo-1735924856823-5c6d23375b1c?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxtaW5pbWFsaXN0JTIwaGFyZSUyMHJhYmJpdCUyMHNpbGhvdWV0dGV8ZW58MXx8fHwxNzU4NzU2OTIxfDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral"
                alt="HareFit Logo"
                className="w-12 h-12 object-contain filter brightness-0 invert"
              />
            </div>
            
            <div className="space-y-4">
              <h1 className="text-4xl md:text-6xl tracking-tight">
                Master Your <span className="text-red-500">Form</span>
              </h1>
              <p className="text-xl text-gray-300 max-w-2xl mx-auto">
                Advanced AI-powered video analysis for perfect exercise technique. 
                Record, upload, and get instant feedback on your workouts.
              </p>
            </div>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button 
                size="lg" 
                className="bg-red-600 hover:bg-red-700 text-white px-8"
              >
                Get Started
                <ArrowRight className="w-4 h-4 ml-2" />
              </Button>
              <Button 
                size="lg" 
                variant="outline" 
                className="border-white/20 text-white hover:bg-white/10 px-8"
              >
                <Play className="w-4 h-4 mr-2" />
                Watch Demo
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* Main Content */}
      <section className="py-16 px-4">
        <div className="container mx-auto space-y-16">
          {/* Interactive Card Deck */}
          <div className="space-y-8">
            <div className="text-center">
              <h2 className="text-white mb-4">Choose Your Path to Perfect Form</h2>
              <p className="text-muted-foreground max-w-2xl mx-auto">
                Select a card to explore different ways to analyze and improve your workout technique
              </p>
            </div>
            
            <CardDeck>
              <VideoRecordSection />
              <FileUploadSection />
              <FitnessJourneySection />
            </CardDeck>
          </div>
          
          {/* Features */}
          <FeaturesSection />
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-white/10 py-8 px-4 mt-16">
        <div className="container mx-auto text-center text-gray-400">
          <p>&copy; 2025 HareFit Analysis. Revolutionizing fitness through AI.</p>
        </div>
      </footer>
    </div>
  );
}