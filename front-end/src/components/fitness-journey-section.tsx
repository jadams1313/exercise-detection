import { useState } from 'react';
import { Button } from './ui/button';
import { Card } from './ui/card';
import { Progress } from './ui/progress';
import { Trophy, Calendar, Target, TrendingUp, ArrowRight, Flame } from 'lucide-react';

export function FitnessJourneySection() {
  const [currentStreak, setCurrentStreak] = useState(12);
  const [weeklyGoal, setWeeklyGoal] = useState(75); // percentage
  const [totalWorkouts, setTotalWorkouts] = useState(48);
  const [monthlyTarget, setMonthlyTarget] = useState(20);

  const achievements = [
    { icon: Trophy, label: 'First Perfect Form', unlocked: true },
    { icon: Flame, label: '7-Day Streak', unlocked: true },
    { icon: Target, label: 'Monthly Goal', unlocked: false },
    { icon: TrendingUp, label: 'Consistency Master', unlocked: false },
  ];

  const handleViewJourney = () => {
    console.log('Redirecting to detailed fitness journey...');
    // This would redirect to the Duolingo-esque progress page
  };

  return (
    <Card className="bg-card border-white/10 p-8">
      <div className="text-center space-y-6">
        <div className="w-16 h-16 bg-red-600/20 rounded-full flex items-center justify-center mx-auto">
          <Trophy className="w-8 h-8 text-red-400" />
        </div>
        
        <div>
          <h3 className="text-white mb-2">My Fitness Journey</h3>
          <p className="text-muted-foreground">
            Track your progress and celebrate milestones on your path to perfect form
          </p>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-2 gap-4 mb-6">
          <div className="bg-black/20 rounded-lg p-4 border border-white/5">
            <div className="flex items-center gap-2 mb-1">
              <Flame className="w-4 h-4 text-orange-400" />
              <span className="text-xs text-muted-foreground">Current Streak</span>
            </div>
            <p className="text-2xl text-white">{currentStreak}</p>
            <p className="text-xs text-orange-400">days</p>
          </div>
          
          <div className="bg-black/20 rounded-lg p-4 border border-white/5">
            <div className="flex items-center gap-2 mb-1">
              <Calendar className="w-4 h-4 text-blue-400" />
              <span className="text-xs text-muted-foreground">This Month</span>
            </div>
            <p className="text-2xl text-white">{totalWorkouts}</p>
            <p className="text-xs text-blue-400">workouts</p>
          </div>
        </div>

        {/* Weekly Progress */}
        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <span className="text-sm text-white">Weekly Goal</span>
            <span className="text-sm text-muted-foreground">{weeklyGoal}%</span>
          </div>
          <Progress value={weeklyGoal} className="h-2" />
          <p className="text-xs text-muted-foreground">5 of 7 workouts completed this week</p>
        </div>

        {/* Achievements Preview */}
        <div className="space-y-3">
          <h4 className="text-white text-sm">Recent Achievements</h4>
          <div className="grid grid-cols-2 gap-2">
            {achievements.slice(0, 4).map((achievement, index) => (
              <div
                key={index}
                className={`flex items-center gap-2 p-2 rounded-lg border ${
                  achievement.unlocked
                    ? 'bg-yellow-500/10 border-yellow-500/30'
                    : 'bg-gray-500/10 border-gray-500/20'
                }`}
              >
                <achievement.icon 
                  className={`w-4 h-4 ${
                    achievement.unlocked ? 'text-yellow-400' : 'text-gray-500'
                  }`} 
                />
                <span className={`text-xs ${
                  achievement.unlocked ? 'text-yellow-200' : 'text-gray-400'
                }`}>
                  {achievement.label}
                </span>
              </div>
            ))}
          </div>
        </div>

        <Button
          onClick={handleViewJourney}
          className="bg-red-600 hover:bg-red-700 text-white w-full"
        >
          View Full Journey
          <ArrowRight className="w-4 h-4 ml-2" />
        </Button>
      </div>
    </Card>
  );
}