import { Card } from './ui/card';
import { Target, TrendingUp, Users, Award } from 'lucide-react';

const features = [
  {
    icon: Target,
    title: 'Precision Analysis',
    description: 'AI-powered motion tracking for perfect form analysis',
  },
  {
    icon: TrendingUp,
    title: 'Progress Tracking',
    description: 'Monitor improvements over time with detailed metrics',
  },
  {
    icon: Users,
    title: 'Expert Feedback',
    description: 'Get professional insights from certified trainers',
  },
  {
    icon: Award,
    title: 'Achievement System',
    description: 'Unlock milestones and celebrate your fitness journey',
  },
];

export function FeaturesSection() {
  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-white mb-4">Why Choose HareFit Analysis?</h2>
        <p className="text-muted-foreground max-w-2xl mx-auto">
          Our cutting-edge technology combines AI motion analysis with expert guidance 
          to revolutionize your fitness journey
        </p>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {features.map((feature, index) => (
          <Card 
            key={index}
            className="bg-card/30 backdrop-blur-sm border-white/10 p-6 text-center hover:bg-card/50 transition-colors"
          >
            <div className="w-12 h-12 bg-red-600/20 rounded-full flex items-center justify-center mx-auto mb-4">
              <feature.icon className="w-6 h-6 text-red-400" />
            </div>
            <h3 className="text-white mb-2">{feature.title}</h3>
            <p className="text-muted-foreground text-sm">{feature.description}</p>
          </Card>
        ))}
      </div>
    </div>
  );
}