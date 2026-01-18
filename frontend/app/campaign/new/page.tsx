"use client"
import * as React from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Loader2, Upload } from "lucide-react"

export default function NewCampaignPage() {
  const [file, setFile] = React.useState<File | null>(null)
  const [name, setName] = React.useState("")
  const [loading, setLoading] = React.useState(false)
  
  // Manual Entry State
  const [manualName, setManualName] = React.useState("Oday")
  const [manualPhone, setManualPhone] = React.useState("+962795910089")
  const [manualItems, setManualItems] = React.useState("Burger Meal")
  const [manualTime, setManualTime] = React.useState("14:00")
  
  const [mode, setMode] = React.useState("upload") // upload | manual

  const router = useRouter()

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFile(e.target.files[0])
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!name) return
    if (mode === "upload" && !file) return

    setLoading(true)
    
    try {
      let res;
      const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8765';
      
      if (mode === "upload") {
        const formData = new FormData()
        if (file) formData.append("file", file)
        
        res = await fetch(`${API_URL}/upload-leads?campaign_name=${encodeURIComponent(name)}`, {
          method: "POST",
          body: formData,
        })
      } else {
        // Manual Entry
        res = await fetch(`${API_URL}/api/campaigns/manual`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            name: name,
            leads: [{
              name: manualName,
              phone: manualPhone,
              items: manualItems,
              time: manualTime
            }]
          }),
        })
      }
      
      if (!res.ok) throw new Error("Failed to create campaign")
      
      const data = await res.json()
      router.push(`/campaign/${data.campaign_id}`)
    } catch (err) {
      console.error(err)
      alert("Error creating campaign")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container mx-auto p-10 max-w-2xl">
      <Card>
        <CardHeader>
          <CardTitle>Create New Campaign</CardTitle>
          <CardDescription>Upload your customer list or add a single test lead.</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="space-y-2">
              <label className="text-sm font-medium">Campaign Name</label>
              <Input 
                placeholder="e.g. Lunch Delivery Jan 15" 
                value={name} 
                onChange={e => setName(e.target.value)} 
                required 
              />
            </div>
            
            <Tabs defaultValue="upload" value={mode} onValueChange={setMode} className="w-full">
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="upload">Upload CSV/Excel</TabsTrigger>
                <TabsTrigger value="manual">Manual Entry</TabsTrigger>
              </TabsList>
              
              <TabsContent value="upload" className="space-y-4 pt-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Customer List File</label>
                  <div className="border-2 border-dashed rounded-lg p-6 flex flex-col items-center justify-center cursor-pointer hover:bg-muted/50 transition-colors">
                    <Input 
                      type="file" 
                      accept=".csv,.xlsx,.xls" 
                      onChange={handleFileChange} 
                      className="hidden" 
                      id="file-upload"
                      required={mode === "upload"}
                    />
                    <label htmlFor="file-upload" className="cursor-pointer flex flex-col items-center">
                      <Upload className="h-10 w-10 text-muted-foreground mb-2" />
                      <span className="text-sm text-muted-foreground">
                        {file ? file.name : "Click to upload Excel or CSV"}
                      </span>
                    </label>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Required columns: Name, Phone (+962...), Items, Time
                  </p>
                </div>
              </TabsContent>
              
              <TabsContent value="manual" className="space-y-4 pt-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Customer Name</label>
                    <Input value={manualName} onChange={e => setManualName(e.target.value)} required={mode === "manual"} />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Phone Number</label>
                    <Input value={manualPhone} onChange={e => setManualPhone(e.target.value)} required={mode === "manual"} />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Items</label>
                    <Input value={manualItems} onChange={e => setManualItems(e.target.value)} required={mode === "manual"} />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Delivery Time</label>
                    <Input value={manualTime} onChange={e => setManualTime(e.target.value)} required={mode === "manual"} />
                  </div>
                </div>
              </TabsContent>
            </Tabs>

            <Button type="submit" className="w-full" disabled={loading}>
              {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
              Create Campaign
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  )
}
