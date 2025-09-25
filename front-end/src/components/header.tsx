import { ImageWithFallback } from './figma/ImageWithFallback';

export function Header() {
  return (
    <header className="border-b border-white/10 bg-black/50 backdrop-blur-sm">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-red-600 rounded-lg flex items-center justify-center">
            <ImageWithFallback 
              src="https://images.unsplash.com/photo-1735924856823-5c6d23375b1c?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxtaW5pbWFsaXN0JTIwaGFyZSUyMHJhYmJpdCUyMHNpbGhvdWV0dGV8ZW58MXx8fHwxNzU4NzU2OTIxfDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral"
              alt="Hare Logo"
              className="w-6 h-6 object-contain filter brightness-0 invert"
            />
          </div>
          <div>
            <h1 className="text-white tracking-tight">HareFit Analysis</h1>
            <p className="text-sm text-red-400">Advanced Exercise Motion Tracking</p>
          </div>
        </div>
      </div>
    </header>
  );
}