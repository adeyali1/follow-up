"use client"
import * as React from "react"
import Link from "next/link"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { PhoneCall, CheckCircle, XCircle, Play, FileText, Loader2, Plus } from "lucide-react"
import { supabase } from "@/lib/supabase" // Need to create this

export default function Dashboard() {
  const [stats, setStats] = React.useState({
    total: 0,
    confirmed: 0,
    cancelled: 0,
    activeCampaigns: 0
  })
  const [campaigns, setCampaigns] = React.useState([])
  const [loading, setLoading] = React.useState(true)

  React.useEffect(() => {
    fetchData()
  }, [])

  async function fetchData() {
    try {
      // Fetch Campaigns
      const { data: campaignsData } = await supabase
        .from('campaigns')
        .select('*')
        .order('created_at', { ascending: false })
      
      setCampaigns(campaignsData || [])

      // Fetch Stats (simplified aggregation for now)
      const { data: leads } = await supabase.from('leads').select('call_status')
      
      const total = leads?.length || 0
      const confirmed = leads?.filter(l => l.call_status === 'CONFIRMED').length || 0
      const cancelled = leads?.filter(l => l.call_status === 'CANCELLED').length || 0
      const active = campaignsData?.filter(c => c.status === 'active').length || 0

      setStats({ total, confirmed, cancelled, activeCampaigns: active })
    } catch (e) {
      console.error(e)
    } finally {
      setLoading(false)
    }
  }

  if (loading) return <div className="flex justify-center p-10"><Loader2 className="animate-spin" /></div>

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold">Kawkab AI Dashboard</h1>
        <Link href="/campaign/new">
          <Button><Plus className="mr-2 h-4 w-4" /> New Campaign</Button>
        </Link>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatsCard title="Total Calls" value={stats.total} icon={<PhoneCall className="h-4 w-4 text-blue-500" />} />
        <StatsCard title="Confirmed" value={stats.confirmed} icon={<CheckCircle className="h-4 w-4 text-green-500" />} />
        <StatsCard title="Cancelled" value={stats.cancelled} icon={<XCircle className="h-4 w-4 text-red-500" />} />
        <StatsCard title="Active Campaigns" value={stats.activeCampaigns} icon={<Play className="h-4 w-4 text-orange-500" />} />
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Recent Campaigns</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="relative w-full overflow-auto">
            <table className="w-full caption-bottom text-sm">
              <thead className="[&_tr]:border-b">
                <tr className="border-b transition-colors hover:bg-muted/50 data-[state=selected]:bg-muted">
                  <th className="h-12 px-4 text-left align-middle font-medium text-muted-foreground">Name</th>
                  <th className="h-12 px-4 text-left align-middle font-medium text-muted-foreground">Status</th>
                  <th className="h-12 px-4 text-left align-middle font-medium text-muted-foreground">Created At</th>
                  <th className="h-12 px-4 text-right align-middle font-medium text-muted-foreground">Action</th>
                </tr>
              </thead>
              <tbody className="[&_tr:last-child]:border-0">
                {campaigns.map((c: any) => (
                  <tr key={c.id} className="border-b transition-colors hover:bg-muted/50">
                    <td className="p-4 align-middle font-medium">{c.name}</td>
                    <td className="p-4 align-middle">
                      <span className={`px-2 py-1 rounded-full text-xs ${c.status === 'active' ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'}`}>
                        {c.status}
                      </span>
                    </td>
                    <td className="p-4 align-middle">{new Date(c.created_at).toLocaleDateString()}</td>
                    <td className="p-4 align-middle text-right">
                      <Link href={`/campaign/${c.id}`}>
                        <Button variant="outline" size="sm">View Report</Button>
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

function StatsCard({ title, value, icon }: any) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        {icon}
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{value}</div>
      </CardContent>
    </Card>
  )
}
